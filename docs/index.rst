Truss Analysis 2D Documentation
================================

Welcome to the **Truss Analysis 2D** API documentation. This package provides a comprehensive toolkit for linear 2D truss finite-element analysis with temperature-dependent steel properties according to EN 1993-1-2 (Eurocode 3).

.. image:: images/logo.png
   :alt: Truss Analysis Logo
   :align: center
   :width: 200px

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install truss_analysis

Basic usage example:

.. code-block:: python

   from truss_analysis import run, Node, Element
   
   # Define nodes
   nodes = [
       Node(id="N1", x=0.0, y=0.0, is_support=True, support_dx=True, support_dy=True),
       Node(id="N2", x=3.0, y=0.0, is_support=True, support_dy=True),
       Node(id="N3", x=1.5, y=2.0),
   ]
   
   # Define elements
   elements = [
       Element(id="E1", node_i="N1", node_j="N3", E=210e9, A=0.01),
       Element(id="E2", node_i="N2", node_j="N3", E=210e9, A=0.01),
       Element(id="E3", node_i="N1", node_j="N2", E=210e9, A=0.01),
   ]
   
   # Define loads
   loads = [{"node_id": "N3", "Fx": 0.0, "Fy": -10000.0}]
   
   # Run analysis
   result = run(nodes, elements, loads)
   print(result.summary())

Key Features
------------

* **Linear 2D Truss Analysis**: Complete finite-element analysis pipeline
* **Temperature-Dependent Properties**: EN 1993-1-2 compliant steel material models
* **Criticality Analysis**: Member importance ranking using exact rank-1 updates
* **Reliability Analysis**: Monte Carlo simulation with uncertainty propagation
* **Retrofit Strategies**: Decision support for structural improvement
* **Topology Generation**: Parametric Warren, Pratt, and Howe truss families
* **Graph Validation**: Structural integrity checks before analysis

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   theory.md
   error_codes.md

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   
   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Development
   
   CONTRIBUTING.md

Core Modules
------------

The package is organized into the following main modules:

Analysis Core
~~~~~~~~~~~~~
* :mod:`truss_analysis.model` - Data structures for nodes and elements
* :mod:`truss_analysis.assembly` - Global matrix assembly
* :mod:`truss_analysis.solver` - Linear system solver with conditioning checks
* :mod:`truss_analysis.postprocess` - Force and reaction calculations
* :mod:`truss_analysis.main` - High-level analysis orchestration

Material & Sections
~~~~~~~~~~~~~~~~~~~
* :mod:`truss_analysis.material.steel_eurocode` - EN 1993-1-2 material properties
* :mod:`truss_analysis.sections` - Cross-section models (idealized HSS)

Advanced Analysis
~~~~~~~~~~~~~~~~~
* :mod:`truss_analysis.criticality` - Member criticality indices and ranking
* :mod:`truss_analysis.limitstates` - Force-based limit states at elevated temperature
* :mod:`truss_analysis.degradation` - Stiffness degradation operators
* :mod:`truss_analysis.reliability` - Monte Carlo reliability engine
* :mod:`truss_analysis.uncertainty` - Random variables and sampling
* :mod:`truss_analysis.retrofit` - Retrofit decision support

Utilities
~~~~~~~~~
* :mod:`truss_analysis.topology_generator` - Parametric truss generation
* :mod:`truss_analysis.graph_validation` - Topology validation
* :mod:`truss_analysis.visualization` - Plotting utilities
* :mod:`truss_analysis.exceptions` - Exception hierarchy
* :mod:`truss_analysis.fileio` - JSON input/output
* :mod:`truss_analysis.units` - Unit conversion

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

This project is licensed under the MIT License - see the :doc:`../LICENSE` file for details.
