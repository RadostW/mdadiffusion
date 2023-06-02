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
    # Compute effective hydrodynamic size of flexible chain of 4 indentical beads
    
    import mdadiffusion

    sizes_four = np.array([1,1,1,1])

    def test_hydrosize():
        testsize = mdadiffusion.mda.mda(sizes_four)
        assert np.allclose(testsize, XXXXX)

    if __name__ == "__main__":
        test_hydrosize()
