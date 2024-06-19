# Cholesterol-related sterols as predictors of COVID-19 disease progression

Author: Sonja Katz (sonja.katz@wur.nl), Miha Moškon (miha.moskon@fri.uni-lj.si), Žiga Pušnik (ziga.pusnik@fri.uni-lj.si)

Wageningen University and Research, Netherlands; Faculty of Computer and Information Science, University of Ljubljana, Slovenia


*31.03.2023*

- Collaboration with Eva Kočar (eva.kocar@mf.uni-lj.si)
- Prediction of COVID-19 disease severity using clinical variables and sterol measurements upon admission (n=164)


# Installation

```bash
conda env create -f environment.yml
source activate env_covid
```

## Publication

[![DOI](https://zenodo.org/badge/621744167.svg)](https://zenodo.org/doi/10.5281/zenodo.12167402)

[Kočar E, Katz S, Pušnik Ž, Bogovič P, Turel G, Skubic C, Režen T, Strle F, Dos Santos VA, Mraz M, Moškon M. COVID-19 and cholesterol biosynthesis: Towards innovative decision support systems. Iscience. 2023 Oct 20;26(10).](https://www.cell.com/iscience/pdf/S2589-0042(23)01876-X.pdf)


![Graphical abstract](https://github.com/sonjakatz/covid_sterols_ML/blob/master/graphicalAbstract.png)



With COVID-19 becoming endemic, there is a continuing need to find biomarkers characterizing the disease and aiding in patient stratification. We studied the relation between COVID-19 and cholesterol biosynthesis by comparing 10 intermediates of cholesterol biosynthesis during the hospitalization of 164 patients (admission, disease deterioration, discharge) admitted to the University Medical Center of Ljubljana. 
The concentrations of zymosterol, 24-dehydrolathosterol, desmosterol, and zymostenol were significantly altered in COVID-19 patients. We further developed a predictive model for disease severity based on clinical parameters alone and their combination with a subset of sterols. Our machine learning models applying 8 clinical parameters predicted disease severity with excellent accuracy (AUC = 0.96), showing substantial improvement over current clinical risk scores. After including sterols, model performance remained better than COVID-GRAM. 
This is the first study to examine cholesterol biosynthesis during COVID-19 and shows that a subset of cholesterol-related sterols is associated with the severity of COVID-19.


## Data

... is available upon reasonable request; point of contact: damjana.rozman@mf.uni-lj.si
