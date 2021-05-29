---
title: "Data Visualisation Post: The Ebb and Flow of A Battle Royale Game: A Visual Guide"
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

## Data Source

My thanks to user KP who scraped the original data and provided it on [Kaggle](https://www.kaggle.com/skihikingkevin/pubg-match-deaths#erangel.jpg). I’ll let them introduce the datasets:

“This dataset provides two zips: aggregate and deaths.

In “deaths”, the files record every death that occurred within the 720k matches. That is, each row documents an event where a player has died in the match.
In “aggregate”, each match’s meta information and player statistics are summarized (as provided by pubg). It includes various aggregate statistics such as player kills, damage, distance walked, etc as well as metadata on the match itself such as queue size, fpp/tpp, date, etc.”

Since there is so much data, I’ve used tiny (1%) random samples in places for clearer visualisation. I grouped my findings into 3 buckets: 1) Survival and Placements, 2) Locations and 3) Weapons and Kills.

## Where Can You Find All The Code?

 All the feature engineering, wrangling and visualisation code can be found at my [Github](https://github.com/Taimur-Shabbir/Battle-Royale-Strategy-Visualisation-and-Analysis/blob/master/Notebook.ipynb).

# Survival and Placements

## The Eternal Debate: To Camp or Not to Camp?

Playstyles in BR games can be imperfectly divided into two categories: i) Camping, where a player/team hides in an advantageous position, waits for opposing players to kill each other and engages only when enemies approach them or ii) Aggression, where players don’t stay in one position for long and may actively seek out enemies.

<img src="{{site.url}}/assets/img/dd.jpg">

Because this is a descriptive analysis and not an experiment, we can’t infer causality. But not shying away from enemy engagements is certainly associated with better in-game outcomes:

<img src="{{site.url}}/assets/img/output_56_0.png">

This graph displays the mean survival time and mean placement by each value in number of kills by individual players.

In other words, every single player that killed two enemies in a match survived, either alone if they were playing alone or in a team, an average of a little over 1000 seconds. Additionally, they placed on average in the late 10s. This measure suggests that better players don’t simply wait for opposing players to kill each other and then engage with the leftovers.

But isn’t it possible that camping players simply pick off enemies while cocooned in their advantageous position?

We can answer this by looking at this question from another angle that captures the essence of camping perhaps more precisely: movement

<img src="{{site.url}}/assets/img/output_60_0.png">

There’s a weak-to-moderate linear relationship where the data is coloured sailor blue.

Interestingly, there are so many instances of where teams/players don’t move much but consistently rank between 15 and 30. However, as placement improves further, distance travelled undeniably increases.

Sticking around in one place may: i) invite successive waves of enemies because they may have found your exact location or ii) not be conducive in a game where the safe zone changes like it does.

So the imperfect answer is that while camping can get you pretty far, you’re more likely to go all the way if you play less like a couch potato.
That leads to another question.

### What’s the most pacifist one can be and still be in with a good statistical chance of winning?

Camping isn’t exactly the same as pacifism, but I wanted to know how few enemies one could kill and still have a decent chance of remaining the last man standing.
Let’s look at all players who won their matches and how many players they killed to get there:

<img src="{{site.url}}/assets/img/output_64_0.png">

50% of winners kill between 4 and 8 players while anything north of 14 kills can be considered pretty extreme.

This seems to be an ideal mix of the two playstyles mentioned to strive towards.

# Locations

### What are the most lethal pockets of Erangel?

Erangel is one of two maps on which data is provided and it is, according to [PUBG](https://pubgmap.io/compare.html), 8 km x 8 km in area. Here are the hotspots on Erangel where most players meet their collective demise:

<img src="{{site.url}}/assets/img/output_21_1.png">

And the original image for comparison:

<img src="{{site.url}}/assets/img/fn.png">


Pochinki is statistically one of the worst places to be on Erangel. It has a geographical neighbour to the south-east that is even less conducive to survival and should be avoided.

I’m not as familiar with PUBG as I am with Warzone but I guess Pochinki and its cousin must: i) offer some good weapons so players are attracted to them or ii) be a common site of the final increments of the safe zone.

Similarly, south-east of Severny should be a no-go zone.

These 3 areas are the most lethal pockets of Erangel, and we could say the same to a lesser extent for the areas north of Primorsk, Mylta and the north-east corner of Military Base.

### When are players most likely to be killed?

I wanted to find an answer to this question that didn’t rely on a time variable but rather on an in-game event, given that the time a match lasts can vary considerably so there are no fixed benchmarks.

‘Placement’ is the rank of a player for a single match. If a player is the first to be eliminated, their placement is equal to the number of players in the game; if n = 100, the player’s placement is 100 (i.e. the worst).

The second player to be eliminated will have a placement of n-1, or 99 in this case.

The winner(s) always has a placement of 1, since they are the last man/team standing. In other words, the lower your placement, the better you played.
Interestingly, players die most often soon after they have just finished pumping lead into a rival. We can come to this conclusion by comparing the placements of Killers and their Victims:

<img src="{{site.url}}/assets/img/output_25_0.png">

The majority of the data is between a difference of -5 and 10 in placement.

Restricted to positive differences, this means that a player who has just killed another is often going to attain a placement that is just <10 places better than than their Victim.
Moreover, Killers follow Victims out of the game almost immediately (0< placement difference < 5) very frequently.

This can likely be due to the mechanic of ‘looting’; when you kill a player in a BR game, you can pick up or loot their equipment. Other players know this and often aim at fresh corpses to pick off would-be vultures.

Conversely, how can we explain the presence of negative placement differences?

The answer is that players may play as a team. If a player kills another player, but the first player’s entire team is eliminated before the second player’s entire team, then the placement difference will be negative for that observation.

# Weapons and Kills

### What weapons are most effective?

Competitive players pretty much do anything to find an edge over others, so the extent to which a weapon is used is a good proxy for its effectiveness.

Here’s an unexpected finding: there isn’t a single sniper rifle among weapons responsible for the most kills across both maps, which is odd considering their sizes. On the other hand, the apex predator is unsurprisingly the assault rifle:

<img src="{{site.url}}/assets/img/output_33_0.png">


The M416 boasts the most kills and is considered a very ‘meta’ weapon in Warzone also. A single submachine gun in the form of the UMP9 sneaks in, which also makes sense as SMGs excel only in close quarters.

It is accompanied in engagements of this nature by shotguns, 3 of which make the cut.

Finally, a possible proxy for the sniper rifle, the marksman rifle, makes its presence known too.

I’m going to give the sniper rifle another chance to shine by stacking the odds in its favour. To do that I need to see the range of kills on Erangel specifically.

### At what range do most conclusive firefights take place on Erangel?

The range of axes in the data is 0, 800000 and, as I mentioned earlier, Erangel is a 64 km² square-shaped map.

This means that 100 ‘steps’ in my ‘Range of Kill’ variable is equal to 1 meter of distance on the ground. I kept this convention instead of converting the distance units to meters.

<img src="{{site.url}}/assets/img/output_39_0.png">

Half of all kills involve the Killer and Victim being between 5 and 75 meters apart with all outliers being north of around 200 meters in distance.

If we expect sniper rifles to excel at long ranges, then we can look for the deadliest weapons at > 80 meters, since this is the approximate 3rd quartile of the above distribution and can serve as a useful benchmark.

But PUBG players really don’t like this class of weapons:

<img src="{{site.url}}/assets/img/output_48_0.png">

Either sniper rifles have too high of skill floor and there aren’t enough players who can use them effectively or players see marksman rifles as better substitutes and use them instead.

Another observation is that far too many players get caught out pretty far from the safe zone.

This is what the Bluezone kills indicate; 5000+ players in this small sample were more than 80 meters away from the safe zone when they died, which is really poor geographical awareness.

It can’t get worse than that, right? Wrong:

<img src="{{site.url}}/assets/img/output_42_0.png">

- Way too many players are asleep at the wheel when they die due to the Bluezone

- Shotguns (S686, S12K, S1897) are effective only at extremely close range

- Most of these outliers are likely freak accidents because bullet drop over long distances is a mechanic replicated in PUBG and very few people can use it effectively
