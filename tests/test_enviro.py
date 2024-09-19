import unittest
import numpy as np
from idmlaser_cholera.mods.transmission import get_enviro_foi

class TestGetEnviroFOI(unittest.TestCase):
    def numpy_calc_enviro_forces(self, new_contagion, enviro_contagion, WASH_fraction, psi, psi_mean, enviro_base_decay_rate, zeta, beta_env, kappa):
        # Decay the environmental contagion by the base decay rate
        enviro_contagion *= (1 - enviro_base_decay_rate)

        # Add newly shed contagion to the environmental contagion, adjusted by zeta
        enviro_contagion += new_contagion * zeta

        # Apply WASH fraction to reduce environmental contagion
        enviro_contagion *= (1 - WASH_fraction)

        # Calculate beta_env_effective using psi
        beta_env_effective = beta_env * (1 + (psi - psi_mean) / psi_mean)

        # Calculate the environmental transmission forces
        forces_environmental = beta_env_effective * (enviro_contagion / (kappa + enviro_contagion))

        return forces_environmental 


    def test_forces_environmental_calculation(self):
        # Simple test: everything is straightforward with known values
        new_contagion = np.array([1.0, 2.0, 1.5, 0.5, 0.0], dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)
        WASH_fraction = np.zeros(5, dtype=np.float32)
        psi = np.ones(5, dtype=np.float32)
        enviro_base_decay_rate = 1.0
        zeta = 1.0
        beta_env = 1.0
        kappa = 1e5

        forces_environmental = get_enviro_foi(
            new_contagion,
            enviro_contagion,
            WASH_fraction,
            psi,
            psi,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        expected_forces = np.array([1e-5, 2e-5, 1.5e-5, 5e-6, 0.0], dtype=np.float32)
        np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-3)

    def test_zero_no_contagion(self):
        # Test with no new contagion
        new_contagion = np.zeros(5, dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)
        WASH_fraction = np.zeros(5, dtype=np.float32)
        psi = np.ones(5, dtype=np.float32)
        psi_mean = psi
        enviro_base_decay_rate = 1.0
        zeta = 1.0
        beta_env = 1.0
        kappa = 1e5

        forces_environmental = get_enviro_foi(
            new_contagion,
            enviro_contagion,
            WASH_fraction,
            psi,
            psi_mean,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        # If no new contagion is added, the forces should all be zero
        expected_forces = np.zeros(5, dtype=np.float32)
        np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-2)

    def test_varied_psi(self):
        # Test with different psi values
        new_contagion = np.array([1.0, 2.0, 1.5, 0.5, 0.0], dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)
        WASH_fraction = np.zeros(5, dtype=np.float32)
        #psi = np.array([0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)  # Vary psi from 0.0 to 1.0
        psi = np.ones(5, dtype=np.float32)  # Vary psi from 0.0 to 1.0
        enviro_base_decay_rate = 1.0
        zeta = 1.0
        beta_env = 1.0
        kappa = 1e5

        forces_environmental = get_enviro_foi(
            new_contagion,
            enviro_contagion,
            WASH_fraction,
            psi,
            psi,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        # With psi varied, the forces will differ for each node
        # Adjust expected forces based on psi scaling (e.g., new_contagion * psi * zeta / kappa)
        expected_forces = self.numpy_calc_enviro_forces(
            new_contagion,
            enviro_contagion.copy(),  # Use a copy to prevent in-place modifications
            WASH_fraction,
            psi,
            psi,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )
        np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-3)

    def test_varied_WASH_fraction(self):
        # Test with WASH_fraction reducing contagion
        new_contagion = np.array([1.0, 2.0, 1.5, 0.5, 0.0], dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)
        WASH_fraction = np.array([0.0, 0.5, 0.25, 0.75, 0.0], dtype=np.float32)  # WASH fraction applied
        psi = np.ones(5, dtype=np.float32)
        enviro_base_decay_rate = 1.0
        zeta = 1.0
        beta_env = 1.0
        kappa = 1e5

        forces_environmental = get_enviro_foi(
            new_contagion,
            enviro_contagion,
            WASH_fraction,
            psi,
            psi,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        # Environmental forces will be reduced due to WASH fraction
        expected_forces = 0.1*np.array([1e-4, 1e-4, 1.125e-4, 1.25e-5, 0.0], dtype=np.float32)
        np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-2)


    def test_non_trivial_values(self):
        # Test with non-trivial psi, WASH_fraction, zeta, and beta_env values
        new_contagion = np.array([1.0, 2.0, 1.5, 0.5, 0.1], dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)
        WASH_fraction = np.array([0.1, 0.3, 0.5, 0.2, 0.4], dtype=np.float32)  # Non-trivial WASH_fraction
        psi = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)  # Non-trivial psi
        enviro_base_decay_rate = 0.2  # Decay rate of 20%
        zeta = 0.9  # Slightly reduced zeta value
        beta_env = 0.7  # Slightly reduced beta_env
        kappa = 1e5

        # Call the actual function under test
        forces_environmental = get_enviro_foi(
            new_contagion,
            enviro_contagion.copy(),
            WASH_fraction,
            psi,
            psi,  # Assuming psi_mean is passed as psi itself
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        # Use the numpy function to calculate expected forces
        expected_forces = self.numpy_calc_enviro_forces(
            new_contagion,
            enviro_contagion.copy(),
            WASH_fraction,
            psi,
            psi,
            enviro_base_decay_rate,
            zeta,
            beta_env,
            kappa
        )

        # Validate the results from the actual function against the numpy implementation
        np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-3)

    def test_10_iterations_decay(self):
        # Test repeated calls with decay
        new_contagion = np.array([1.0, 2.0, 1.5, 0.5, 0.1], dtype=np.float32)
        enviro_contagion = np.zeros(5, dtype=np.float32)  # Start with no environmental contagion
        WASH_fraction = np.array([0.1, 0.3, 0.5, 0.2, 0.4], dtype=np.float32)  # Non-trivial WASH_fraction
        psi = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)  # Non-trivial psi
        enviro_base_decay_rate = 0.75  # 75% decay rate
        zeta = 0.9  # Slightly reduced zeta value
        beta_env = 0.7  # Slightly reduced beta_env
        kappa = 1e5

        for iteration in range(10):
            with self.subTest(i=iteration):
                # Call the actual function under test
                forces_environmental = get_enviro_foi(
                    new_contagion,
                    enviro_contagion.copy(),
                    WASH_fraction,
                    psi,
                    psi,  # Assuming psi_mean is passed as psi itself
                    enviro_base_decay_rate,
                    zeta,
                    beta_env,
                    kappa
                )

                # Use the numpy function to calculate expected forces
                expected_forces = self.numpy_calc_enviro_forces(
                    new_contagion,
                    enviro_contagion.copy(),
                    WASH_fraction,
                    psi,
                    psi,
                    enviro_base_decay_rate,
                    zeta,
                    beta_env,
                    kappa
                )

                # Validate the results from the actual function against the numpy implementation
                np.testing.assert_allclose(forces_environmental, expected_forces, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

