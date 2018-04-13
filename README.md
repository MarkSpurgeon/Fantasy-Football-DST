# Predicting-Fantasy-Football-DST-Weekly-Performance-with-Machine-Learning

## Introduction
Fantasy football involves selecting a set of NFL players based on how well you think they will perform according to a metric (fantasy points) which is a function of real-life player performance. In my time participating in fantasy football I have learned that the system governing fantasy football performance is very complex and extremely difficult to predict. Nevertheless, there are numerous popular tools for informing your decisions on which players to choose for a given week (the functional unit of the NFL season). These tools include expert rankings, algorithms based on known determinants of performance, and basic heuristics for things like "recent performance quality" or "quality of the competition". I have utilized each of these to varying degrees, but remain dissatisfied with how poorly they translate to successful predictions. Therefore, I am interested in building a tool that predicts future fantasy football performance by framing the problem as a learning task.

Management of a fantasy football team involves choosing which players to use at each position in a roster. The DST (defense & special teams) position often has many available choices, meaning that there is a greater potential to make the “correct” choice in a given week. Therefore, accurate prediction of defensive fantasy performances has the potential to influence a team’s fantasy score. This score is the primary metric determining success in fantasy football. To this end I have created models for predicting the rankings of defenses on a weekly basis. I have assessed performance of these models by comparing their predictions to the true outcomes. Lastly, I have created visualizations comparing the performance of my models to other predictions available to the casual fantasy football player. 

## Objectives

(1) Compare multiple machine learning models' predictions to those of an expert and heuristic methods

(2) Identify features that are most important to the models

## The Script

**Scraping:** In its current state, the script scrapes data from 3 sources. In order to allow for the script to run without an internet connection, I have provided the scraped data in downloadable form and modified the script to default to loading this data instead of scraping. In order to initiate scraping, you just need to change the "if False:" statements in the fantasy football stats (line 31), schedule (line 64), and vegas lines (line 136) sections. The expert rankings are not available online in their entirety, so they must be loaded in from a csv file.

**Processing:** The script generates a histogram of all fantasy scores from the season along with their log transform. In this case, the log transform actually makes the data less normally distributed according to KS test (line 359), so the default is to leave the scores unaltered.

**Analysis:** This analysis tries to predict future defensive performances based on information from previous defensive performances. There are basically two choices for how to treat your samples: you can create a model for each of the 32 teams, which could key into team-specific attributes but would suffer from extremely small sample size - or you can create a general model by aggregating all previous defensive performances and hoping to capture a general signal. I have chosen to create a general model because such a model seems like it could be more robust predictively (week-to-week or even year-to-year), as well as generating more interesting insights.
Once sample treatment has been established, we must figure out what form the features will take. Here the features are relatively simple, leaving ample room for expansion in the future. The first 10 features are the average stats for the defensive unit year-to-date. The next set of 10 features are the average state for the defensive unit over the previous 2 weeks. The final 5 features are game spread, game over/under, time of day of the game, whether the team is home or away, and day of week of the game.
Finally, ridge regression and random forest were utilized in order to model fantasy scores from these features. Ridge regression was chosen because it is a relatively simple method that can handle a (reasonably) large number of features including noise features. Random forest was chosen because it is able to capture deep conditional relationships that I believe may exist in this system, and because it is reasonably interpretable.

**Performance Visualization:** To compare performance among methods, Kendall's tau rank correlation coefficient between predicted and true rankings was calculated. In the rank_correlation.pdf file, each point is a correlation score for a given model for a single week (points are slightly jittered along x-dimension to increase visibility of overlapping points). The black lines indicate the average over all 12 tested weeks. The predictons of Dylan Lerch (author of "Defense Wins Championships" on Reddit and Fantasypros.com - @dtlerch on Twitter) are in purple. My machine learning methods are in orange and green. Pink and blue are simple heuristics that would be easily applied by any fantasy football player. Red is a slightly more complicated heuristic that would still be easy to apply by anyone with slightly more effort.

**Feature Visualization:** In order to get a sense of which features were driving the predictions, feature importances were extracted from both models. For ridge regression, importances were the absolute value of model parameters which had been normalized such that they summed to 1. For random forest, importances were derived from the standard variance-reduction metric that had been used to grow the forest. These importances were plotted using stacked bar charts with each stack representing a different week of the season. In this way you can get a sense of which features are most important over all weeks, but also if features vary in importance from week to week.

## Brief Conclusions and Future Work



The primary challenge in this project was and continues to be feature engineering. Even with the simple data that I have scraped, hundreds of features could be produced that would allow the models to utilize different - potentially useful - information. For example, it could be that a defense's most recent 4 games are the most predictive and that games prior to that have little bearing on future performance. In that case, you might want to include the mean fantasy score over previous 4 weeks as a feature. Similar features could be produced for a vast array of scenarios. Perhaps the biggest surprise is that "naive + versus" method performs almost exactly as well as random forest. The "naive + versus" method simply averages the mean fantasy score for a defense and the mean fantasy score against the upcoming opponent. For example, if an upcoming matchup involved Buffalo and Miami, the "naive + versus" method would take the mean fantasy points scored by Buffalo's defense over the entire season and average this number with the mean fantasy points scored by DST units in their matchups with Miami. In doing so, this heuristic attempts to strike a balance between ranking highly defenses that have performed well and ranking highly defenses that have advantageous matchups. The success of "naive + versus" tells me that the next step of this project should involve finding a way to incorporate information about the quality of upcoming opponent.

Random forest comes surprisingly close to matching the performance of the well-respected expert. Since there remains much room for improving the features, it does seem possible that further improvement could allow me to match this expert's performance.

Things that I would be interested in exploring in the future:
**(1)** does incorporating information about quality of opponent improve performance?

**(2)** do the plots look qualitatively similar for the 2016 season? More specifically, do these methods perform the same with respect to each other and do the same features stick out as most important?

**(3)** most ambitiously: are there other extremely important tidbits of information that could be folded into these models? I'm thinking specifically about things like a backup quarterback starting the upcoming game - this is something that is often leveraged by amateurs and experts alike.
