scikit-FIBERS
======================================


Feature Inclusion Bin Evolver for Risk Stratification (FIBERS) (under development, publication forthcoming) is
an evolutionary algorithm for binning features to stratify risk in biomedical datasets.

FIBERS utilizes an evolutionary algorithm approach to optimizing bins of features based on their stratification of event risk through the following steps:

1) Random bin initialization or expert knowledge input; the bin value at an instance is the sum of the instance's values for the features included in the bin
2) Repeated evolutionary cycles consisting of:
   - Candidate bin evaluation with log-rank test to evaluate for significant difference in survival curves of the low risk group (instances for which bin value = 0) and high risk group (instances for which bin value > 0).
   - Genetic operations (elitism, parent selection, crossover, and mutation) for new bin discovery and generation of the next generation of candidate bins
3) Final bin evaluation and summary of risk stratification provided by top bins


Installation
-----------------------------

We can easily install scikit-fibers using the following command:

.. code-block:: bash

    pip install scikit-fibers


Documentation for FIBERS Class:
--------------------------------

Documentation for the FIBERS class can be found `here <skfibers.html#module-skfibers.fibers>`_.

Contact
-------------------------------

Please email sdasariraju23@lawrenceville.org and Ryan.Urbanowicz@cshs.org for any
inquiries related to FIBERS.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents:


   self
   modules

