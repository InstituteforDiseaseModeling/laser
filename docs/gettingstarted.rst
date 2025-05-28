===============
Getting started
===============

Research scientist
------------------

.. csv-table::
   :header: "Task", "Documentation", "Notebook"
   :widths: 15, 15, 15

   "Collect and prepare input data", ":doc:`Demographics <architecture>`; :doc:`User customizability <architecture>` ", "`Outbreak size in the SIR model <https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/04_SIR_nobirths_outbreak_size.ipynb>`_; Average age at infection in the SIR model"
   "Select a model", "SIR; Vital dynamics; SIR-synthetic; SIR-real", "SI Model with no demographics; SIS Model with no demographics "
   "Configure model parameters", "", ""
   "Run simulations", "", ""
   "Visualize and analyze outputs", "", ""
   "Evaluate interventions", "", ""
   "Calibrate the model", "", ""
   "Communicate results", "", ""
   "Iterate", "", ""


Research scientist-developer
----------------------------

.. csv-table::
   :header: "Task", "Documentation", "Notebook"
   :widths: 15, 15, 15

   "Build custom model logic", ":doc:`Demographics <architecture>`; :doc:`User customizability <architecture>` ", "`Outbreak size in the SIR model <https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/04_SIR_nobirths_outbreak_size.ipynb>`_; Average age at infection in the SIR model"
   "Add agent properties", "Application layer; The model object; Components", "SIS Model with no demographics; Outbreak size in the SIR model"
   "Reuse demographics/migration tools", "", ""
   "Parameter calibration", "", ""
   "Combine components modularly", "", ""
   "Extend models with logic", "", ""
   "Add agent properties", "", ""
   "Perform batch runs or sensitivity", "", ""

Software engineer
-----------------

.. csv-table::
   :header: "Task", "Documentation", "API; Files"
   :widths: 15, 15, 15

   "Explore/understand architecture", "", ""
   "Add new algorithms/modules", "", ""
   "Benchmark & optimize", "", ""
   "Use Numba/OpenMP/SIMD", "", ""
   "Write tests", "", ""
   "Add CLI or utils", "", ""
   "Document modules", "", ""
   "Contribute via GitHub", "", ""