# Predicting-Fantasy-Football-DST-Weekly-Performance-with-Machine-Learning

## Introduction
Fantasy football involves selecting a set of NFL players based on how well you think they will perform according to a metric (fantasy points) which is a function of real-life player performance. In my time participating in fantasy football I have learned that the system governing fantasy football performance is very complex and extremely difficult to predict. Nevertheless, there are numerous popular tools for informing your decisions on which players to choose for a given week (the functional unit of the NFL season). These tools include expert rankings, algorithms based on known determinants of performance, and basic heuristics for things like "recent performance quality" or "quality of the competition". I have utilized each of these to varying degrees, but remain dissatisfied with how poorly they translate to successful predictions. Therefore, I am interested in building a tool that predicts future fantasy football performance by framing the problem as a learning task.

Management of a fantasy football team involves choosing which players to use at each position in a roster. The defense position often has many available choices, meaning that there is a greater potential to make the “correct” choice in a given week. Therefore, accurate prediction of defensive fantasy performances has the potential to influence a team’s fantasy score. This score is the primary metric determining success in fantasy football. To this end I have created models for predicting the rankings of defenses on a weekly basis. I have assessed performance of these models by comparing their predictions to the true outcomes. Lastly, I have created visualizations comparing the performance of my models to other predictions available to the casual fantasy football player. 

## Objectives

(1) Compare multiple machine learning models' predictions to those of an expert and heuristic methods

(2) Identify features that are most important to the models
