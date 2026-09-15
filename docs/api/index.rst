API Reference
=============

This section provides comprehensive API documentation for all public modules in the Truss Analysis 2D package. The documentation is automatically generated from docstrings using Sphinx autodoc with NumPy-style formatting.

Quick Navigation
----------------

* :ref:`Core Analysis Modules <core-modules>`
* :ref:`Material & Sections <material-sections>`
* :ref:`Criticality & Limit States <criticality-limitstates>`
* :ref:`Reliability & Uncertainty <reliability-uncertainty>`
* :ref:`Retrofit Analysis <retrofit-analysis>`
* :ref:`Utilities & Helpers <utilities-helpers>`

.. _core-modules:

Core Analysis Modules
---------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.model
   truss_analysis.assembly
   truss_analysis.solver
   truss_analysis.postprocess
   truss_analysis.main
   truss_analysis.fileio

.. _material-sections:

Material & Sections
-------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.material.steel_eurocode
   truss_analysis.sections

.. _criticality-limitstates:

Criticality & Limit States
--------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.criticality.engine
   truss_analysis.criticality.criteria
   truss_analysis.criticality.indices
   truss_analysis.criticality.ranking
   truss_analysis.criticality.scenarios
   truss_analysis.limitstates
   truss_analysis.degradation

.. _reliability-uncertainty:

Reliability & Uncertainty
-------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.reliability
   truss_analysis.uncertainty.random_variables
   truss_analysis.uncertainty.sampling
   truss_analysis.uncertainty.streaming
   truss_analysis.uncertainty.probabilistic_ranking

.. _retrofit-analysis:

Retrofit Analysis
-----------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.retrofit.actions
   truss_analysis.retrofit.strategies
   truss_analysis.retrofit.costs

.. _utilities-helpers:

Utilities & Helpers
-------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.numerics
   truss_analysis.topology_generator
   truss_analysis.graph_validation
   truss_analysis.visualization
   truss_analysis.exceptions
   truss_analysis.units
   truss_analysis.sensitivity
   truss_analysis.heterogeneity
   truss_analysis.thermal.fire_curve
   truss_analysis.thermal.protection
   truss_analysis.thermal.material
   truss_analysis.tangent_verification
   truss_analysis.reliability_adapter

Validation (Optional Dependencies)
----------------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   truss_analysis.validation.metrics
   truss_analysis.validation.opensees_reference
   truss_analysis.validation.surrogate

Complete Module Tree
--------------------

For a complete hierarchical view of all modules, see:

* :doc:`truss_analysis` - Full package tree

Indices
-------

* :ref:`genindex` - Function, class and variable index
* :ref:`modindex` - Module index
* :ref:`search` - Full-text search
