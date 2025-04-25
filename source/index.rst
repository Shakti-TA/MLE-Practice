Welcome to the MLE-Practice Documentation
=========================================

This project focuses on the prediction of the **Median Housing Value** using various machine learning techniques. The housing data can be downloaded from the following URL:

`Housing Data <https://raw.githubusercontent.com/ageron/handson-ml/master/>`_

The script includes code to automatically download the dataset. The project models the **median house value** based on the provided housing data.

Techniques Used
---------------
The following machine learning techniques have been applied:

- Linear Regression
- Decision Tree
- Random Forest

Steps Performed
---------------
The following steps were performed during the project:

1. **Data Preparation and Cleaning:**
   - We clean the data and handle any missing values by checking and imputing them.

2. **Feature Generation and Correlation Check:**
   - Features are generated, and the variables are checked for correlation.

3. **Sampling and Splitting the Dataset:**
   - Multiple sampling techniques are evaluated. The dataset is split into training and test sets.

4. **Modeling and Evaluation:**
   - The above-mentioned modeling techniques (Linear Regression, Decision Tree, and Random Forest) were applied and evaluated.
   - The final evaluation metric used is **mean squared error (MSE)**.

Execution Instructions
----------------------
To execute the script, follow the steps below:

1. **Setup the Environment:**
   - Create the environment using the following command:

   .. code-block:: bash

      conda env create -f env.yml

2. **Run the Script:**
   - After setting up the environment, run the script with the following command:

   .. code-block:: bash

      python3 nonstandardcode.py

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   ingest_data_script
   train_script
   score_script
