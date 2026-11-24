"""Shallow-water solver driver.

Scheme: MUSCL reconstruction of (eta, u, v, b) with minmod, Audusse hydrostatic
reconstruction, HLLC with Einfeldt speeds, SSP-RK2 in time, semi-implicit
Manning friction applied as an operator split after the RK step.

Well-balanced for lake-at-rest including partially dry cells; see
docs/scheme.md for the cancellation and tests/hydro/ for the gates.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from shoreloop.closures import breaking as BR
from shoreloop.closures import foam as FM
from shoreloop.closures import viscosity as V
from shoreloop.closures import wetness as WT
from shoreloop.config import BC, SolverConfig
from shoreloop.device import prepare_device
from shoreloop.numerics import update as U
from shoreloop.numerics.timestep import max_wave_rate
from shoreloop.physics import boundary as B
from shoreloop.physics.state import Scratch, State
from shoreloop.scene.forcing import DeviceForcing, WaveForcing


class SWESolver:
    def __init__(self, cfg: SolverConfig | None = None):
        self.cfg = cfg or SolverConfig()
        g = self.cfg.grid
        self.device = prepare_device(self.cfg.device)
        self.state = State(g, self.cfg.device)
        self.scratch = Scratch(g, self.cfg.device)
        self.steps = 0
        # Launch dimensions used repeatedly.
        self._dim_full = g.shape
        self._dim_cells = (g.nx, g.ny)
        self._dim_faces_x = (g.nx + 1, g.ny)
        self._dim_faces_y = (g.nx, g.ny + 1)
        self._dim_ghost_x = (g.ng, g.shape[1])
        self._dim_ghost_y = (g.shape[0], g.ng)
        p = self.cfg.physics
        bc = self.cfg.bc
        if BC.WAVEMAKER in (bc.xhi, bc.ylo, bc.yhi):
            # The ghost-fill kernels silently treat an unknown kind as OPEN,
            # which is reflective for long waves -- fail loudly instead.
            raise ValueError("BC.WAVEMAKER is only implemented on xlo")
        c = self.cfg.closures
        if c.foam_enabled and not c.breaking_enabled:
            # Foam is sourced only by measured breaking power; without the
            # breaking closure the source pipeline would still have to run,
            # making breaking_enabled=False a silent lie.
            raise ValueError("foam_enabled requires breaking_enabled: foam is "
                             "sourced only by measured breaking dissipation")
        self._viscous = p.nu_const > 0.0 or p.smagorinsky_cs > 0.0
        if self._viscous:
            # Cell-centred eddy viscosity, precomputed so the face coefficient
            # can be a symmetric function of the two adjacent cells (see
            # closures/viscosity.py: a one-sided Smagorinsky nu makes the
            # operator asymmetric and lets it inject energy).
            self.nu_field = wp.zeros(g.shape, dtype=self.state.h.dtype,
                                     device=self.cfg.device)
        self._closures_on = (self.cfg.closures.breaking_enabled
                             or self.cfg.closures.foam_enabled)
        if self._closures_on:
            dev, shape = self.cfg.device, g.shape
            z = lambda: wp.zeros(shape, dtype=self.state.h.dtype, device=dev)
            self.eta_old = z()
            self.E_old = z()
            self.E_new = z()
            self.FEx = z()
            self.FEy = z()
            self.FEx2 = z()
            self.FEy2 = z()
            self.D_raw = z()
            self.wcount = z()
            self.D = z()
            self.P_fric = z()
            self.B = z()
            self.P_break = z()
            self.cf = z()
            self.cf_new = z()
            # [0] dissipated energy, [1] clamped deficit, [2] friction energy
            self.diss_acc = wp.zeros(3, dtype=wp.float64, device=dev)
        self._wetness = self.cfg.closures.wetness_enabled
        if self._wetness:
            self.w = wp.zeros(g.shape, dtype=self.state.h.dtype,
                              device=self.cfg.device)
        self.forcing: DeviceForcing | None = None
        if self.cfg.bc.xlo == BC.WAVEMAKER:
            # Default to a pure absorber so the boundary is well defined even
            # before set_forcing() is called.
            from shoreloop.scene.forcing import still_water
            self.forcing = still_water(0.0).to_device(self.cfg.device, g.Ly)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def set_bathymetry(self, b: np.ndarray) -> None:
        """Set the bed over the interior and fill its ghost cells once.

        The bed is static, so ghost values are computed here rather than every
        step. Scenes that know b analytically may overwrite the padded array
        afterwards.
        """
        self.state.set("b", b)
        bc = self.cfg.bc
        g = self.cfg.grid
        wp.launch(B.bc_scalar_x, dim=self._dim_ghost_x,
                  inputs=[self.state.b, int(bc.xlo), int(bc.xhi), g.ng, g.nx],
                  device=self.device)
        wp.launch(B.bc_scalar_y, dim=self._dim_ghost_y,
                  inputs=[self.state.b, int(bc.ylo), int(bc.yhi), g.ng, g.ny],
                  device=self.device)

    def set_forcing(self, forcing: WaveForcing) -> None:
        """Install offshore wave forcing. Requires bc.xlo == BC.WAVEMAKER.

        Oblique components are validated against the shallow-water dispersion
        relation at the boundary depth: a mode with k_y >= omega / sqrt(g h)
        is evanescent, not oblique -- generating it would force a wave that
        cannot propagate. Call after set_bathymetry, which defines that depth.
        """
        if self.cfg.bc.xlo != BC.WAVEMAKER:
            raise ValueError("set_forcing requires BCConfig(xlo=BC.WAVEMAKER)")
        g = self.cfg.grid
        if any(m != 0 for m in forcing.ky_modes):
            b_col = self.state.get("b")[0, :]
            h_ref = float(forcing.eta0 - b_col.mean())
            if h_ref <= 0.0:
                raise ValueError("oblique forcing needs a wet offshore "
                                 "boundary; call set_bathymetry first")
            c = np.sqrt(self.cfg.physics.g * h_ref)
            for n, m in zip(forcing.harmonics, forcing.ky_modes):
                omega = 2.0 * np.pi * n / forcing.period
                sth = (2.0 * np.pi * abs(m) / g.Ly) * c / omega
                if sth >= 0.95:
                    m_max = int(omega * g.Ly / (2.0 * np.pi * c) * 0.95)
                    raise ValueError(
                        f"harmonic n={n} with ky mode m={m} is evanescent at "
                        f"the boundary depth ({h_ref:.2f} m): sin(theta) = "
                        f"{sth:.2f}. Max |m| for this harmonic is {m_max}.")
        self.forcing = forcing.to_device(self.cfg.device, g.Ly)

    def set_water(self, h: np.ndarray, qx: np.ndarray | None = None,
                  qy: np.ndarray | None = None) -> None:
        z = np.zeros_like(h)
        self.state.set("h", h)
        self.state.set("qx", z if qx is None else qx)
        self.state.set("qy", z if qy is None else qy)

    # ------------------------------------------------------------------
    # pieces of a step
    # ------------------------------------------------------------------
    def apply_bc(self, h, qx, qy, t: float = 0.0) -> None:
        bc, g, p = self.cfg.bc, self.cfg.grid, self.cfg.physics
        wavemaker = bc.xlo == BC.WAVEMAKER
        # bc_x handles both x sides in one launch; when xlo is a wavemaker, fill
        # it as OPEN here and overwrite immediately below.
        kind_lo = int(BC.OPEN) if wavemaker else int(bc.xlo)
        wp.launch(B.bc_x, dim=self._dim_ghost_x,
                  inputs=[h, qx, qy, kind_lo, int(bc.xhi), g.ng, g.nx],
                  device=self.device)
        if wavemaker:
            f = self.forcing
            # Reduce the phase argument mod T on the host in float64 before it
            # is cast to FLOAT for the kernel. Unwrapped, t reaches hundreds of
            # seconds where float32 resolves only ~3e-5 s, so the forcing at
            # t = kT + s stops being the forcing at s and the "exact" loop
            # silently degrades with every period -- the same trap the ripple
            # closure documents and dodges in closures/ripples.py.
            t_red = float(np.fmod(t, f.period)) if f.period > 0.0 else float(t)
            wp.launch(B.bc_wavemaker_x_lo, dim=self._dim_ghost_x,
                      inputs=[h, qx, qy, self.state.b,
                              f.amp, f.harm, f.pha, f.ky, f.nmodes,
                              t_red, f.period, f.eta0,
                              p.g, g.ng, p.desing_eps, p.h_dry,
                              g.y0, g.dy],
                      device=self.device)
        wp.launch(B.bc_y, dim=self._dim_ghost_y,
                  inputs=[h, qx, qy, int(bc.ylo), int(bc.yhi), g.ng, g.ny],
                  device=self.device)

    def compute_dt(self) -> float:
        g, p = self.cfg.grid, self.cfg.physics
        self.scratch.wave_rate.zero_()
        wp.launch(max_wave_rate, dim=self._dim_cells,
                  inputs=[self.state.h, self.state.qx, self.state.qy,
                          self.scratch.wave_rate, g.ng, g.dx, g.dy,
                          p.g, p.desing_eps],
                  device=self.device)
        c = self.cfg.closures
        if c.foam_enabled and c.foam_kappa > 0.0:
            # Explicit foam diffusion has its own limit; fold it into the same
            # conservative sum as the advective rate.
            self._foam_rate = 2.0 * c.foam_kappa * (1.0 / g.dx ** 2 + 1.0 / g.dy ** 2)
        else:
            self._foam_rate = 0.0
        rate = float(self.scratch.wave_rate.numpy()[0]) + self._foam_rate
        if self._viscous:
            # Explicit diffusion adds its own stability limit. Summing the
            # advective and diffusive rates is the conservative combination --
            # which requires a separate reduction slot: max-ing both into one
            # scalar would bound dt by max(adv, visc), allowing a combined
            # Courant number of up to 2x the intended one.
            self.scratch.visc_rate.zero_()
            self._compute_nu(self.state.h, self.state.qx, self.state.qy)
            wp.launch(V.max_viscous_rate, dim=self._dim_cells,
                      inputs=[self.state.h, self.nu_field,
                              self.scratch.visc_rate, g.ng, g.dx, g.dy,
                              p.h_wet_recon],
                      device=self.device)
            rate += float(self.scratch.visc_rate.numpy()[0])
        if rate <= 0.0:
            return p.dt_max
        return min(p.dt_max, p.cfl / rate)

    def _rhs(self, h, qx, qy, t: float = 0.0) -> None:
        """Fill scratch.r* with -div(F) + S_topo for the given state."""
        g, p, s = self.cfg.grid, self.cfg.physics, self.scratch
        self.apply_bc(h, qx, qy, t)
        wp.launch(U.zero3, dim=self._dim_full,
                  inputs=[s.rh, s.rqx, s.rqy], device=self.device)

        for axis, (dim_faces, ds) in enumerate(
            ((self._dim_faces_x, g.dx), (self._dim_faces_y, g.dy))
        ):
            wp.launch(U.reconstruct, dim=self._dim_full,
                      inputs=[h, qx, qy, self.state.b,
                              s.etaL, s.etaR, s.unL, s.unR, s.utL, s.utR,
                              s.bL, s.bR, axis, p.desing_eps, p.h_wet_recon],
                      device=self.device)
            wp.launch(U.compute_faces, dim=dim_faces,
                      inputs=[s.etaL, s.etaR, s.unL, s.unR, s.utL, s.utR,
                              s.bL, s.bR, s.Fh, s.Fn, s.Ft, s.Bs,
                              axis, g.ng, p.g, p.h_dry],
                      device=self.device)
            wp.launch(U.accumulate_divergence, dim=self._dim_cells,
                      inputs=[s.etaL, s.etaR, s.bL, s.bR,
                              s.Fh, s.Fn, s.Ft, s.Bs,
                              s.rh, s.rqx, s.rqy,
                              axis, g.ng, ds, p.g],
                      device=self.device)

        if self._viscous:
            self._compute_nu(h, qx, qy)
            wp.launch(V.add_viscous_momentum, dim=self._dim_cells,
                      inputs=[h, qx, qy, self.nu_field, s.rqx, s.rqy,
                              g.ng, g.dx, g.dy, p.h_wet_recon, p.desing_eps],
                      device=self.device)

    def _compute_nu(self, h, qx, qy) -> None:
        g, p = self.cfg.grid, self.cfg.physics
        wp.launch(V.compute_nu, dim=self._dim_full,
                  inputs=[h, qx, qy, self.nu_field, g.dx, g.dy,
                          p.nu_const, p.smagorinsky_cs, p.desing_eps],
                  device=self.device)

    def _clamp(self, h, qx, qy) -> None:
        g = self.cfg.grid
        wp.launch(U.clamp_depth, dim=self._dim_cells,
                  inputs=[h, qx, qy, self.scratch.clipped, g.ng, g.cell_area],
                  device=self.device)

    # ------------------------------------------------------------------
    # the step
    # ------------------------------------------------------------------
    def step(self, dt: float | None = None) -> float:
        g, p, s, st = self.cfg.grid, self.cfg.physics, self.scratch, self.state
        if dt is None:
            dt = self.compute_dt()

        if self._closures_on:
            self._closure_pre()

        # stage 1: U1 = U^n + dt L(U^n)
        t_n = st.t
        self._rhs(st.h, st.qx, st.qy, t_n)
        wp.launch(U.euler_stage, dim=self._dim_cells,
                  inputs=[st.h, st.qx, st.qy, s.rh, s.rqx, s.rqy,
                          s.h1, s.qx1, s.qy1, dt, g.ng],
                  device=self.device)
        self._clamp(s.h1, s.qx1, s.qy1)

        # stage 2: U^{n+1} = 1/2 U^n + 1/2 (U1 + dt L(U1))
        # The wavemaker is time dependent, so the second stage must be forced at
        # t + dt, not t, or the boundary is only first-order accurate in time.
        self._rhs(s.h1, s.qx1, s.qy1, t_n + dt)
        wp.launch(U.ssp_rk2_combine, dim=self._dim_cells,
                  inputs=[st.h, st.qx, st.qy, s.h1, s.qx1, s.qy1,
                          s.rh, s.rqx, s.rqy, st.h, st.qx, st.qy, dt, g.ng],
                  device=self.device)
        self._clamp(st.h, st.qx, st.qy)

        # friction: operator split, semi-implicit, cannot reverse the velocity
        if p.friction_enabled and p.manning_n > 0.0:
            wp.launch(U.apply_manning_friction, dim=self._dim_cells,
                      inputs=[st.h, st.qx, st.qy, dt, p.g, p.manning_n,
                              p.desing_eps, p.h_wet_recon, g.ng],
                      device=self.device)

        if self._closures_on:
            # Refresh the ghost cells of the FINAL state before any closure
            # reads them. Without this, the closure energy fluxes here (and in
            # _closure_pre at the start of the next step) see ghosts left over
            # from the last _rhs call -- one stage stale. At a wall a fresh
            # mirror ghost gives face velocity (-u+u)/2 = 0 exactly; a stale
            # one gives (u_new - u_old)/2 != 0, i.e. a spurious energy flux
            # through solid rock, polluting the measured dissipation over the
            # whole boundary ring.
            self.apply_bc(st.h, st.qx, st.qy, t_n + dt)
            self._closure_step(dt)

        if self._wetness:
            c = self.cfg.closures
            wp.launch(WT.advance_wetness, dim=self._dim_cells,
                      inputs=[self.w, st.h, dt, c.tau_wet, c.tau_dry,
                              p.h_wet_recon, g.ng],
                      device=self.device)

        st.t += dt
        self.steps += 1
        return dt

    # ------------------------------------------------------------------
    def _closure_pre(self) -> None:
        """Snapshot eta and energy, and the energy flux, before the step."""
        g, p, st = self.cfg.grid, self.cfg.physics, self.state
        wp.launch(BR.store_eta, dim=self._dim_full,
                  inputs=[st.h, st.b, self.eta_old], device=self.device)
        wp.launch(BR.energy_density, dim=self._dim_full,
                  inputs=[st.h, st.qx, st.qy, st.b, self.E_old, p.g,
                          p.desing_eps], device=self.device)
        # Pre-step half of the trapezoidal energy flux; _closure_step computes
        # the post-step half. The trapezoid matters: a forward-Euler flux
        # against an SSP-RK2 state update injects a spurious O(dt) oscillating
        # residual at every bore (docs/decisions.md, "Breaking").
        wp.launch(BR.energy_flux, dim=(g.nx + 1, g.ny + 1),
                  inputs=[st.h, st.qx, st.qy, st.b, self.FEx, self.FEy,
                          p.g, p.desing_eps, g.ng], device=self.device)

    def _closure_step(self, dt: float) -> None:
        g, p, c, st = (self.cfg.grid, self.cfg.physics, self.cfg.closures,
                       self.state)
        wp.launch(BR.energy_density, dim=self._dim_full,
                  inputs=[st.h, st.qx, st.qy, st.b, self.E_new, p.g,
                          p.desing_eps], device=self.device)
        wp.launch(BR.energy_flux, dim=(g.nx + 1, g.ny + 1),
                  inputs=[st.h, st.qx, st.qy, st.b, self.FEx2, self.FEy2,
                          p.g, p.desing_eps, g.ng], device=self.device)
        wp.launch(BR.measure_dissipation, dim=self._dim_cells,
                  inputs=[self.E_old, self.E_new, self.FEx, self.FEy,
                          self.FEx2, self.FEy2,
                          st.h, st.qx, st.qy, self.D_raw, self.P_fric,
                          self.diss_acc, dt, g.dx, g.dy, p.g,
                          p.manning_n if p.friction_enabled else 0.0,
                          p.h_wet_recon, p.desing_eps, g.ng, g.cell_area],
                  device=self.device)
        wp.launch(BR.wet_stencil_count, dim=self._dim_full,
                  inputs=[st.h, self.wcount, p.h_wet_recon, g.ng, g.nx, g.ny],
                  device=self.device)
        wp.launch(BR.smooth_and_clamp, dim=self._dim_cells,
                  inputs=[self.D_raw, st.h, self.wcount, self.D,
                          self.diss_acc, dt,
                          p.h_wet_recon, g.ng, g.cell_area],
                  device=self.device)
        wp.launch(BR.breaking_indicator, dim=self._dim_cells,
                  inputs=[st.h, st.b, self.eta_old, self.B, dt, p.g,
                          c.breaking_gamma, c.breaking_smooth,
                          p.h_wet_recon, g.ng], device=self.device)
        wp.launch(BR.gate_breaking_power, dim=self._dim_cells,
                  inputs=[self.D, self.B, self.P_break, g.ng],
                  device=self.device)
        if c.foam_enabled:
            # Foam is carried state and its stencil reads the halo, so it needs
            # ghost cells like any other transported field. Without this the
            # halo stays zero forever: foam crossing the alongshore periodic
            # seam is silently destroyed (a deletion stripe at y = 0), and the
            # diffusion term drains foam at every edge. WAVEMAKER falls through
            # to OPEN in the scalar fill, which is the right behaviour for a
            # passive field at a radiating boundary.
            bc = self.cfg.bc
            wp.launch(B.bc_scalar_x, dim=self._dim_ghost_x,
                      inputs=[self.cf, int(bc.xlo), int(bc.xhi), g.ng, g.nx],
                      device=self.device)
            wp.launch(B.bc_scalar_y, dim=self._dim_ghost_y,
                      inputs=[self.cf, int(bc.ylo), int(bc.yhi), g.ng, g.ny],
                      device=self.device)
            wp.launch(FM.advance_foam, dim=self._dim_cells,
                      inputs=[self.cf, self.cf_new, st.h, st.qx, st.qy,
                              self.P_break, dt, g.dx, g.dy,
                              c.foam_alpha, c.foam_tau, c.foam_kappa,
                              c.foam_max, p.h_wet_recon, p.desing_eps, g.ng],
                      device=self.device)
            self.cf, self.cf_new = self.cf_new, self.cf

    # ------------------------------------------------------------------
    def field(self, name: str) -> np.ndarray:
        """Interior of a closure diagnostic field as a host array."""
        g = self.cfg.grid
        sl = (slice(g.ng, g.ng + g.nx), slice(g.ng, g.ng + g.ny))
        return getattr(self, name).numpy()[sl]

    def run(self, t_end: float, dt: float | None = None,
            callback=None, callback_every: int = 0) -> None:
        """Advance to `t_end`, landing exactly on it.

        The remaining interval is split into equal steps no larger than the
        CFL step rather than clipping the last one: a clipped sliver can be
        arbitrarily small, and the breaking closure divides float32 field
        differences by dt, so a ~1e-9 s step turns roundoff into an enormous
        spurious dissipation spike.
        """
        while self.state.t < t_end - 1e-12:
            step_dt = self.compute_dt() if dt is None else dt
            remaining = t_end - self.state.t
            if dt is None:
                n = max(1, int(np.ceil(remaining / step_dt - 1e-9)))
                step_dt = remaining / n
            else:
                step_dt = min(step_dt, remaining)
            self.step(step_dt)
            if callback is not None and callback_every and self.steps % callback_every == 0:
                callback(self)

    # ------------------------------------------------------------------
    @property
    def dissipated_energy(self) -> float:
        """Cumulative energy removed by breaking, in m^4/s^2 (i.e. J/rho)."""
        return float(self.diss_acc.numpy()[0])

    @property
    def dissipation_deficit(self) -> float:
        """Cumulative negative residual clamped away: truncation error."""
        return float(self.diss_acc.numpy()[1])

    @property
    def friction_energy(self) -> float:
        return float(self.diss_acc.numpy()[2])

    @property
    def clipped_mass(self) -> float:
        """Cumulative water volume invented by the h >= 0 clamp, in m^3."""
        return float(self.scratch.clipped.numpy()[0])

    def reset_clipped(self) -> None:
        self.scratch.clipped.zero_()
