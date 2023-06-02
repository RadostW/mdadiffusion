MDA Diffusion documentation
===========================

.. toctree::
   :hidden:
   :maxdepth: 1

Minimum Dissipation Approximation is an improved approximation to hydrodynamic radius computation (improved over Kirkwood-Riesmann).
With this package you can compute hydrodynamic size of IDPs using SARW spheres as conformation generator and PyGRPY as mobiltiy 
tensors approximation.


How to install
''''''''''''''

.. prompt:: bash $ auto

  $ python3 -m pip install mdadiffusion

and you'll be good to go.

Package contents
''''''''''''''''

.. automodule:: mdadiffusion.mda
   :members:

   
Example use
'''''''''''

.. prompt:: python >>> auto

    # Copyright (C) Radost Waszkiewicz 2023
    # This software is distributed under MIT license
        
    import pytest

    import os
    import numpy as np
    import yaml

    import mdadiffusion


    @pytest.mark.datafiles("./tests/default_config.yaml")
    def test_mda_pipeline(datafiles):
        path = str(datafiles)
        with open(os.path.join(path, "default_config.yaml")) as cfg_file:
            config = yaml.safe_load(cfg_file)

        bead_model = mdadiffusion.mda.bead_model_from_sequence(
            annotated_sequence="GPSAGLVPRGSGG[IEGRHMLEEIWDVQDIPPSMQAQMH]SHGTQSSSSSSSSSSSSSNGSSNGNSSSNSNSSQHGPHPHPHGQQLTPNQQQHQQQHSQLQQVHANGSGSGGGSNNNSSSGGVVPGLGMLDQV",
            effective_density=config["OrderedBeads"]["EffectiveDensity"],
            hydration_thickness=config["OrderedBeads"]["HydrationThickness"],
            disordered_radii=config["DisorderedBeads"]["HydrodynamicRadius"],
            c_alpha_distance=config["DisorderedBeads"]["CAlphaDistance"],
            aa_masses=config["AminoAcidMasses"],
        )

        assert np.allclose(
            bead_model["bead_description_compact"],
            [[4.2, 1.9025, 13], [14.09435, 14.09435, 1], [4.2, 1.9025, 93]],
        )

        rh_dict = mdadiffusion.mda.hydrodynamic_size(
            bead_steric_radii=bead_model["steric_radii"],
            bead_hydrodynamic_radii=bead_model["hydrodynamic_radii"],
            ensemble_size=100,
            bootstrap_rounds=20,
        )

        rtol = 0.05
        assert np.allclose(rh_dict["rh_mda"] , 27.55196, rtol = rtol)
        assert np.allclose(rh_dict["rh_kr"] , 24.4554, rtol = rtol)
