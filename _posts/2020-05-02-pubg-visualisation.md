---
title: "Data Visualisation Post - The Ebb and Flow of A Battle Royale Game: A Visual Guide"
date: 2020-05-02
tags: [Data Visualisation, videogames, strategy, visualisation]
#header:
image: j.jpg
#mathjax: "true"
---

There’s nothing like an epidemic to keep us glued to our screens. In time for the coronavirus-induced quarantine, game developer Infinity Ward released Warzone, a Battle Royale (BR) game mode, for Call of Duty: Modern Warfare, a game with a 50 million player count. I’ve been playing with some friends during the lockdown and wanted to find out what changes we could make in our gameplay to s̶t̶o̶p̶ ̶b̶e̶i̶n̶g̶ ̶b̶l̶i̶n̶d̶s̶i̶d̶e̶d̶ ̶a̶n̶d̶ ̶c̶o̶n̶s̶i̶s̶t̶e̶n̶t̶l̶y̶ ̶m̶o̶w̶e̶d̶ ̶d̶o̶w̶n̶ ̶i̶n̶ ̶o̶u̶r̶ ̶r̶u̶s̶h̶ ̶t̶o̶ ̶g̶e̶t̶ ̶t̶o̶ ̶t̶h̶e̶ ̶s̶a̶f̶e̶ ̶z̶o̶n̶e̶ ̶t̶o̶w̶a̶r̶d̶s̶ ̶t̶h̶e̶ ̶e̶n̶d̶ ̶o̶f̶ ̶e̶a̶c̶h̶ ̶m̶a̶t̶c̶h̶ improve overall.

I didn’t find any data on Warzone but I did find loads of data on PUBG, a BR game that shares most mechanics with Warzone and can, in fact, be considered the latter’s progenitor.

My subsequent analysis using Seaborn did provide me with gameplay tips but it broadly became a data-driven look at the general in-game behaviour of players and how it related to their survival and winning chances.

If you want to stop being a liability to your teammates or understand why your significant other spends hours on these games and justifies it by extolling their complexity and need for strategy, read on.

## Short primer on BR games:

50–100 players, either playing alone or in teams, are airdropped onto a large map without any weapons or equipment. The aim of the game is to collect these items from the map and be the last person/team standing by killing rival players.

The mechanic of a ‘safe zone’ ensures the match keeps progressing. A safe zone is circular in shape and shrinks in increments over time. If players are caught outside of this safe zone, they die after a short while. This way, surviving players are eventually corralled into a tiny safe area towards the end of the game, where they have no choice but to kill each other.

Every increment of the safe zone is a) randomly decided but b) a subset of the previous increment, so there is an element of controlled unpredictability that makes gameplay and strategy a little unique.

## Data and Code

My thanks to user KP who scraped the original data and provided it on [Kaggle](https://www.kaggle.com/skihikingkevin/pubg-match-deaths#erangel.jpg). I’ll let them introduce the datasets:

“This dataset provides two zips: aggregate and deaths.

In “deaths”, the files record every death that occurred within the 720k matches. That is, each row documents an event where a player has died in the match.
In “aggregate”, each match’s meta information and player statistics are summarized (as provided by pubg). It includes various aggregate statistics such as player kills, damage, distance walked, etc as well as metadata on the match itself such as queue size, fpp/tpp, date, etc.”

Since there is so much data, I’ve used tiny (1%) random samples in places for clearer visualisation. I grouped my findings into 3 buckets: 1) Survival and Placements, 2) Locations and 3) Weapons and Kills. Finally, all the feature engineering, wrangling and visualisation code can be found at my Github in a Jupyter Notebook.

# Survival and Placements

## The Eternal Debate: To Camp or Not to Camp?

Playstyles in BR games can be imperfectly divided into two categories: i) Camping, where a player/team hides in an advantageous position, waits for opposing players to kill each other and engages only when enemies approach them or ii) Aggression, where players don’t stay in one position for long and may actively seek out enemies.

<img src="{{ site.url }}{{ site.baseurl }}//img/dd.jpg" alt="linearly separable data">
