---
layout: post
title: "Online physics homework help: what, when and how?"
date: 2019-11-12
thumbnail: https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/20191117_thumb.png
---

<p align="center"><img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/20191117_thumb.png" style="width: 300px" class="border"></p>

<div class="fineprint">The Python code used for this post can be found here: <a href="https://github.com/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191117/physics_help_analysis.py" target="_blank">https://github.com/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191117/physics_help_analysis.py</a></div>

<div class="fineprint">Images and tables are generally scrollable if not displayed in full.</div>

As a physics teacher who grew up on the internet, I'm well aware of the many resources available to students online. One key aspect of this is the availability of help through forums (and more recently through social media or homework help apps).

These forums are a huge source of data! I wanted to find out what topics students were asking about, and discern patterns about their asking habits.

For this attempt, I used data from <a href="https://www.physicsforums.com/forums/introductory-physics-homework-help.153/" target="_blank">https://www.physicsforums.com/forums/introductory-physics-homework-help.153/</a>. Physicsforums.com is one of the biggest and oldest still-active forums, and quite often, when searching for information about particular high-school to undergrad topics in physics, I see posts from this forum turn up among the top search results. I decided to focus only on the "introductory physics homework help" section since the terms are likely to be less complex to analyse than higher level content - especially the use of formula and symbols.

Ah, yes. Formula and symbols. That was possibly the worst part of this analysis.

Over three days, I scraped all the data that I could get hold of from the forum. For some reason my script missed out a lot of the data from 2012 onwards, so I decided to just settle for the earliest decade of data for now to get some preliminary results. I used all the data I had (i.e. all 2003 - 2012, and the sparse data from 2012 - 2019) to do topic modeling, but only analysed the threads from 2003 - 2012.

Let's talk about cleaning. Usually, this isn't a very difficult process: remove typos, websites, maybe names, things that aren't going to be relevant to the analysis process. But in physics (and chemistry or math) you can expect to see a lot of single letters and symbols. Is a standalone "k" a typo? Or does it refer to the spring constant, or some other coefficient? It's really hard to tell. That's why I limited my raw data to just introductory physics, since the range of common uses for each symbol is reduced. Furthermore, I'd expect most students to just try to follow common conventions for symbols (e.g. <i>m</i> for mass, <i>a</i> for acceleration, and write Newton's 2nd Law as "<i>F</i>=<i>ma</i>").

When trying to assign labels for the topics I obtained through topic modeling, I tried to match letters (or combinations of letters - e.g. based on the context of the other words in a topic, I assumed "ma" to refer to Newton's 2nd Law rather than someone's mother). This took quite some effort and I had to think quite hard about the syllabus I had taught for several years. Surprisingly, it's easy to recognise and use symbols in context, but when you're staring at a collection of words and symbols, it can sometimes take a bit of effort to guess what they mean.

However, something rather nice turned up. I was worried about the use of greek letters and special symbols; however, this forum integrates <a href="https://en.wikibooks.org/wiki/LaTeX" target="_blank">TeX</a>, and so my results were peppered with words like "lambda", "sqrt", or "cdot" (these are the greek letter λ, square root, and · symbol to denote multiplication respectively).

Furthermore, while doing my scraping, I didn't find any convenient way to remove some advertisments between posts. I cut out all the other parts of the page, but every thread had a 'post' between two actual posts that was simply a link to three other threads on the forum that they decided was interesting. I removed some of the more obvious words from my analysis (in particular, "birdsong", "el nino", "warming", "acid" were advertisments that appeared on a huge number of my posts). Some of these were harder to remove, though - for example, "birdsong" is clearly not a common physics term, but "plates" could be commonly used in physics. I didn't remove these yet, and they ended up being attributed to a few topics. I can then just ignore these topics in my later analysis.

Well, that's enough about the text analysis. I tried, but frankly the data here will be slightly noisy. Here's the outcome of the topic modeling:

<a href="https://nbviewer.jupyter.org/github/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191117/pyLDAvis_physicsforums_topics.ipynb" target="_blank">View online</a>

<a href="https://github.com/not-even-wong/not-even-wong.github.io/blob/master/_posts/20191117/lda60.html" target="_blank">Download</a>

These are the topics that I identified, along with the most relevant terms at λ = 0.6:


<div style="height: 300px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Topic</th>
<th title="Field #2">Label</th>
<th title="Field #3">Most relevant terms</th>
</tr></thead>
<tbody><tr>
<td align="right">1</td>
<td>Data analysis</td>
<td>graph, line, measur, lab, uncertainti, vs, curv, error, data, slope, plot, straight, tangent, experiment, time</td>
</tr>
<tr>
<td align="right">2</td>
<td>Newtons laws</td>
<td>law, newton, nd, rd, second, st, applic, appli, tricki, action, k, use, f=ma, coulomb, plate</td>
</tr>
<tr>
<td align="right">3</td>
<td>Sound waves</td>
<td>wave, frequenc, sound, hz, wavelength, vibrat, stand, fundament, amplitude, string, hear, effect, phase, open, pipe</td>
</tr>
<tr>
<td align="right">4</td>
<td>Work-energy theorem</td>
<td>work, w,j, check, theorem, joul, sub, require, displac, k, power, kg, someon, total, distanc</td>
</tr>
<tr>
<td align="right">5</td>
<td>Thermodynamic cycles</td>
<td>Delta, ideal, expans, thermodynam, process, engine, effici, mole, pv, cycl, volum, temperature, pressur, revers, p</td>
</tr>
<tr>
<td align="right">6</td>
<td>Tension</td>
<td>tension, string, rope, centripet, circl, motion, cabl, tangenti, circular, hang, uniform, cord, radial, suspend, ceil</td>
</tr>
<tr>
<td align="right">7</td>
<td>Fluid pressure</td>
<td>pressur,water, densiti, volum, fluid, rho, atmosphere, depth, liquid, pipe, float, flow, principl, tube, air</td>
</tr>
<tr>
<td align="right">8</td>
<td>Friction</td>
<td>friction, , coeffici, static, normal, mu, crate, kinet, fn, box, horizont, μ, surface, block, appli, push</td>
</tr>
<tr>
<td align="right">9</td>
<td>Electromagnetic induction</td>
<td>wire, loop, magnet, current, coil, induc, emf, induct, flux, electromagnet, carri, b, strength, turn, direct</td>
</tr>
<tr>
<td align="right">10</td>
<td>Electric field</td>
<td>q, coulomb, electrostat, ring, e, kq, net, magnitude, point, corner, distanc, strength, place, attract, locat</td>
</tr>
<tr>
<td align="right">11</td>
<td>Conservation of momentum</td>
<td>momentum, collis, conserve, impuls, elast, inelast, mv, collid, kg, impact, veloc, final, vf, mass, befor</td>
</tr>
<tr>
<td align="right">12</td>
<td>Distance and displacement</td>
<td>km, ft, displac, north, east, wind, hr, west, south, hour, walk, h, feet, fli, min</td>
</tr>
<tr>
<td align="right">13</td>
<td>Projectile motion (2D)</td>
<td>projectile, motion, horizont, height, launch, rang, maximum, max, vy, vx, vo, initi, trajectory, angl, land</td>
</tr>
<tr>
<td align="right">14</td>
<td>Quantum</td>
<td>e, electron, atom, proton, photon, hydrogen, decay, quantum, state, mc, number, emit, level, energi, model</td>
</tr>
<tr>
<td align="right">15</td>
<td>Math: geometry</td>
<td>p, point, o, intersect, lowest, line, locat, diagram, expand, right, upper, let, distanc, highest, shown</td>
</tr>
<tr>
<td align="right">16</td>
<td>Heat capacity</td>
<td>water, ice, heat, specif, cube, temperature, capac, c, kg, temp, degree, gram, j, ml, liquid</td>
</tr>
<tr>
<td align="right">17</td>
<td>Circuits - capacitance</td>
<td>capacitor, capacit, parallel, signific, digit, c, store, connect, plate, seri, round, fig, voltag, switch, figur</td>
</tr>
<tr>
<td align="right">18</td>
<td>Remove: irrelevant</td>
<td>slow, k, molecule, natur, crack, shed, toughest, exist, build, avoid, destruct, materi, studi, slip, plate</td>
</tr>
<tr>
<td align="right">19</td>
<td>Moment of inertia</td>
<td>rotat, angular, inertia, rod, moment, disk, axi, mr, cylind, omega, linear, spin, translat, rad, ω</td>
</tr>
<tr>
<td align="right">20</td>
<td>Centre of gravity</td>
<td>mass, center, kg, gravity, centr, dimension, ratio, analysi, distanc, locat, edg, hole, total, far, man</td>
</tr>
<tr>
<td align="right">21</td>
<td>Heat transfer rates</td>
<td>increas, heat, temperature, rate, thermal, transfer, decreas, loss, room, cool, conduct, radiat, absorb, flow, temp</td>
</tr>
<tr>
<td align="right">22</td>
<td>Math: Logarithm</td>
<td>u, log, motor, ln, let, arrang, mu, rais, use, e, step, convers, substitute, concern, plz</td>
</tr>
<tr>
<td align="right">23</td>
<td>Filler: school</td>
<td>test, class, exam, im, student, realli, math, ani, like, school, grade, idea, veri, hard, probabl</td>
</tr>
<tr>
<td align="right">24</td>
<td>Math: proportionality</td>
<td>g, constant, ’, proport, relationship, invers, vari, directli, depend, expand, express, increase, non, assum, decreas</td>
</tr>
<tr>
<td align="right">25</td>
<td>Rocket motion</td>
<td>veloc, average, rocket, initi, final, instantan, time, given, constant, s, zero, instant, interv, displac, dure</td>
</tr>
<tr>
<td align="right">26</td>
<td>Special relativity</td>
<td>particl, rel, frame, observ, transform, gamma, special, relativist, stationari, time, respect, rest, measur, travel, simultan</td>
</tr>
<tr>
<td align="right">27</td>
<td>Math: vectors</td>
<td>vector, compon, angl, degre, magnitud, triangl, direct, axi, perpendicular, °, result, draw, hypotenus, y, diagram</td>
</tr>
<tr>
<td align="right">28</td>
<td>Optics (geometric)</td>
<td>object, imag, len, refract, mirror, ray, optic, reflect, glass, index, incid, light, distanc, size, angl</td>
</tr>
<tr>
<td align="right">29</td>
<td>Oscillation</td>
<td>pendulum, simpl, harmon, period, amplitud, motion, shm, omega, pi, oscil, swing, maximum, phase, π, frequenc</td>
</tr>
<tr>
<td align="right">30</td>
<td>Drag force</td>
<td>bodi, fall, free, air, drag, diagram, resist, termin, object, rigid, graviti, draw, act, downward, neglect</td>
</tr>
<tr>
<td align="right">31</td>
<td>Gauss&#39;s law</td>
<td>surfac, shell, insid, sphere, spheric, conductor, conduct, distribut, inner, gauss, outer, flux, cylind, densiti, metal</td>
</tr>
<tr>
<td align="right">32</td>
<td>Velocity</td>
<td>s, time, kinemat, distanc, second, travel, sec, vi, vf, d, meter, use, long, far, veloc</td>
</tr>
<tr>
<td align="right">33</td>
<td>Orbits</td>
<td>earth, orbit, gravit, planet, satellit, sun, moon, star, gm, revolut, angular, rad, radian, univers, period</td>
</tr>
<tr>
<td align="right">34</td>
<td>Remove: advertisment</td>
<td>k, plate, link, block, frequent, warm, extrem, identifi, dynam, copper, polici, eletromagnet, gr, attitud, lift</td>
</tr>
<tr>
<td align="right">35</td>
<td>Kinematics: solving for constant acceleration</td>
<td>t, time, equat, vt, function, s, plug, second, given, initi, deriv, express, solut, eq, set</td>
</tr>
<tr>
<td align="right">36</td>
<td>Circuits: ohm&#39;s law</td>
<td>circuit, current, voltag, resist, resistor, ohm, batteri, seri, parallel, flow, connect, sourc, equival, ac, termin</td>
</tr>
<tr>
<td align="right">37</td>
<td>Springs</td>
<td>spring, compress, kx, stretch, hook, machin, elast, constant, equilibrium, k, releas, block, maximum, attach, mass</td>
</tr>
<tr>
<td align="right">38</td>
<td>Math: symbols for dirac notation</td>
<td>vec, z, h, hat, cdot, partial, epsilon, notat, j, oper, vector, element, phi, complex, prove</td>
</tr>
<tr>
<td align="right">39</td>
<td>Remove: irrelevant</td>
<td>doc, al, bullet, expand, block, realiz, youll, right, wrong, analyz, use, figur, sinc, correct, shot</td>
</tr>
<tr>
<td align="right">40</td>
<td>Projectiles</td>
<td>ball, ground, drop, hit, thrown, rock, stone, height, throw, jump, bounc, floor, fall, cliff, upward</td>
</tr>
<tr>
<td align="right">41</td>
<td>Math: symbols for coordinate systems</td>
<td>lambda, coordin, dot, polar, λ, central, orient, d, rectangular, cylindr, origin, cross, axe, circl, express</td>
</tr>
<tr>
<td align="right">42</td>
<td>Length measurements</td>
<td>cm, unit, length, area, mm, diamet, meter, dimens, section, cross, convers, convert, l, steel, multipli</td>
</tr>
<tr>
<td align="right">43</td>
<td>Math: curl</td>
<td>left, right, tini, tim, hand, rule, ​pole, direct, clockwis, counter, expand, arrow, middl, page</td>
</tr>
<tr>
<td align="right">44</td>
<td>Conservation of energy</td>
<td>energi, kinet, ke, conserv, pe, mv, mgh, mechan, total, potenti, h, height, lost, initi, gravit</td>
</tr>
<tr>
<td align="right">45</td>
<td>Math: calculus</td>
<td>x, y, function, dx, dy, xy, ax, deriv, axi, differenti, integr, limit, sqrt, calculu, calc</td>
</tr>
<tr>
<td align="right">46</td>
<td>Math: differentiation</td>
<td>frac, integr, dt, int, dv, dx, pi, differenti, ln, deriv, partial, limit, right, arrow, left, omega</td>
</tr>
<tr>
<td align="right">47</td>
<td>Math: series expansion</td>
<td>n, sum, seri, number, right, prove, th, limit, expand, sigma, altern, proof, total, use, infin</td>
</tr>
<tr>
<td align="right">48</td>
<td>Electric potential</td>
<td>potenti, differ, infin, region, point, zero, volt, electrostat, e, bring, higher, distanc, energi, separ, gravit</td>
</tr>
<tr>
<td align="right">49</td>
<td>Vehicles</td>
<td>car, truck, train, stop, road, hill, brake, tire, quick, track, drive, deceler, travel, curv, turn</td>
</tr>
<tr>
<td align="right">50</td>
<td>Filler: questions and clarifications</td>
<td>whi, expand, say, mean, case, onli, zero, true, doesnt, happen, like, way, ha, depend, differ</td>
</tr>
<tr>
<td align="right">51</td>
<td>Math: greek symbols</td>
<td>theta, sin, l, θ, tan, phi, angl, trig, pi, sine, ident, arc, cosin, radian, sec</td>
</tr>
<tr>
<td align="right">52</td>
<td>Math: algebra</td>
<td>equat, use, squar, formula, im, root, tri, expand, unknown, variabl, plug, algebra, number, wrong, solut</td>
</tr>
<tr>
<td align="right">53</td>
<td>Math: sign of scalar values</td>
<td>neg, sign, direct, half, opposit, convent, minu, life, alway, indic, magnitud, mean, quantiti, sinc, whi</td>
</tr>
<tr>
<td align="right">54</td>
<td>Remove: advertisment</td>
<td>earthquak, crosstalk, board, doubl, tube, astronuc, twice, end, cut, k, singl, close, plate, width, setup</td>
</tr>
<tr>
<td align="right">55</td>
<td>MCQ choices</td>
<td>b, c, d, ab, choic, multipl, ac, e, follow, ii, correct, begin, check, right equal</td>
</tr>
<tr>
<td align="right">56</td>
<td>Attached file</td>
<td>f, reaction, attach, kb, jpg, solut, statement, expand, equat, tag, identifi, extrem, frequent, warm, correct</td>
</tr>
<tr>
<td align="right">57</td>
<td>Inclined plane problems</td>
<td>inclin, plane, ramp, cart, roll, slope, tabl, frictionless, block, hw, slide, horizont, parallel, degre, k</td>
</tr>
<tr>
<td align="right">58</td>
<td>Diffraction</td>
<td>power, light, intens, interfer, slit, wavelength, diffract, nm, sourc, grate, radiat, phase, minimum, red, shift</td>
</tr>
<tr>
<td align="right">59</td>
<td>Torque</td>
<td>torqu, alpha, equilibrium, wheel, wall, beam, arm, support, bridg, pivot, balanc, static, bar, stick, moment</td>
</tr>
<tr>
<td align="right">60</td>
<td>Pulleys</td>
<td>pulley, weight, net, mg, kg, box, ma, elev, act, pull, exert, upward, block, downward, normal</td>
</tr>
<tr>
<td align="right">61</td>
<td>Rotational motion</td>
<td>angular, omega, rotat, alpha, rad, spin, ω, revolut, radian, round, linear, tangenti, sec, w, min</td>
</tr>
</tbody></table>
</div>

You might have noticed that topic 61 doesn't appear in the pyLDAvis visualisation. That's because it actually comes from a different model. I had some challenges when deciding on the number of topics for my topic modeling:
<ul>
<li>For lower numbers of topics, each time I re-ran it (starting with a different random seed) I got different (but relevant) topics.</li>
<li>When increasing the number of topics, more of these relevant topics started to show up, but I could not get ALL of them to show up at once.</li>
<li>On the other hand, increasing the number of topics also increased the number of 'junk' meaningless topics.</li>
</ul>

So I had a dilemma. I wanted to increase the number of topics to get all the possible relevant topics to appear at once. However, the amount of nonsense also increased. The particular run I've uploaded here has 56 relevant topics out of 60, and has almost all the topics that I identified as relevant from previous rounds...except "Rotational motion" (which I think got absorbed into some of the related topics). Since this data is primarily for me to reduce the dimensionality of the original data and identify trends in the posts, I decided that it would be ok if I used this set of 60 topics, ignored the 4 irrelevant topics, and added 1 topic from another model. Of course, the weights won't be balanced properly, but since I was going to normalise all the weights anyway, and I'm not comparing the proportion of weights <i>for each individual document</i> I think that's fine.

Looking at these topics, it's surprising how cleanly they correspond to actual physics topics in the syllabus. In fact, I've noticed that when using a smaller number of topics for the model, I get more general topics from the syllabus, and when increasing the number of topics, they do split nicely into actual subtopics from the syllabus. This isn't surprising since a lot of the actual topics from the syllabus have similar terms. On the other hand, that also makes it harder to use latent dirichlet allocation (LDA) to generate topic models.

Out of these topics, they can broadly be split into physics knowledge/techniques (n=40), math skills or notation (n=13), filler (n=4) and advertisments/irrelevant (n=4).

The filler category is particuarly interesting:
<ul>
<li>Multiple choice options (A, B, C, D, and other related terms).</li>
<li>Attached files and formats.</li>
<li>Requests for clarification (e.g. "why", "say", "mean", etc.)</li>
<li>Talking about school (e.g. "test", "exam", "class").</li>
</ul>

I'm actually really surprised that these came up as topics of their own! I guess that shows how significant and relevant these are when high school students ask for homework help online.

<b>Objective 1: determine factors affecting thread views or replies</b>

Since I only had data from 2003 to 2012, and I scraped the data in 2019, it becomes hard to see long term trends. I suspect that thread views and replies tend to level off over time, so all posts past a certain age become stagnant. This is reflected in the following scatterplots:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/plot%20replies%20against%20age.png">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/plot%20views%20against%20age.png">

That said, you can see that in the early days of the forum, activity took a while to ramp up. I attribute this to a smaller forum population during the first few years leading to less views and replies.

There are some threads with an exceptionally large number of views. The top 5 are:
<ul>
  <li>converting km/hr to m/s</li>
  <li>tension formula</li>
  <li>need egg drop project ideas</li>
  <li>find spring constant of spring in n/m help</li>
  <li>phase difference mutual inductor</li>
</ul>

The first 4 are indeed incredibly common questions. I can see why they'd have the most views if they turn up quite high up in google results - it's a vicious cycle once it hits critical mass.

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/google%20screenshot.png">

I also attempted to deduce a relationship between views and replies:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/correlation%20-%20views%20against%20replies.png">

Well... that doesn't look very good. My theory is that replies definitely plateau quite early since there's only so much that can be said constructively about a particuarly question. However, even after replies have stopped, the thread will continue to acquire views, as long as it stays relevant on Google searches.

Then, what about the relationship between thread topic and thread views or replies?

Each thread has weights for each topic indicating how strongly that thread relates to that topic. I tried to correlate each topic's weight against the views and replies. The result? The largest <i>R<sup>2</sup></i> is 0.42, followed by 0.29, and vast majority of topics pretty much have negligible amounts. It's not hard to see why, when every single scatterplot of views or replies against topic weight looks something like this:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/correlation%20-%20replies%20against%20gauss%20law.png">

What about correlations between topics? After removing the advertisments, this is what the correlation matrix between topic weights looks like:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/correlation%20remove%20spam.png">

and the "most correlated" topics have scatterplots that look like these:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/correlation%20-%20rotational%20motion%20against%20moment%20of%20inertia.png">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/correlation%20-%20differentiation%20against%20calculus.png">

So it seems like the topic weights don't really affect thread popularity, but more surprisingly, I couldn't find any strong correlation between different topics. I would have thought that some topics would tend to crop up together (e.g. a student talking about electric fields would be likely to also talk about electroc potentials). My suggested explanation for this lack of trend is that while this may be true for more general discussions of a topic in the syllabus, on the homework help forum, students tend to ask isolated questions (e.g. "How do I calculate the electric field of this setup?"). Therefore, the results wouldn't show a strong relationship between different topics.

<b>Objective 2: observe trends for topics over time</b>

My second goal was to determine whether any trends can be seen over time for each topic. Hence, I aggregated the average weight of each topic for each month, and then plotted these average weights over time... 

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/topic%20weights%20over%20time.png">
<div class="fineprint">(each of the 60 topics has a slightly different colour; it would be impossible to deduce relationships directly from this graph, but it's meant to give a quick overview to try to spot trends)</div>

It doesn't look like there are any long term trends. But wait! Some of those lines look like they're periodic in nature! 
The blue, green, and purple data with the largest average weights are quite clearly periodic, and if you look carefully, other lines look periodic too.

I aggregated the average weight of each topic by month (i.e. all Januaries, all Februaries, etc.) and plotted them. To make the trends more obvious, I normalised them between 0 to 2 and then squared the values, so the peaks will stand out more:

<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/topic%20weights%20by%20month.png">

That definitely looks periodic. I've also uploaded all the individual (un-normalised) graphs to this album for you to peruse at your casual pleasure: <a href="https://imgur.com/a/lmNukMr">https://imgur.com/a/lmNukMr</a>

I would attribute these periodic trends to the time of the year that it is being taught. I expect to see at least two peaks, one for schools in English-speaking countries where the school year starts in August, and another for those starting in January. This is true for a lot of thee graphs.

To test this suggestion, I tried doing clustering <i><b>based on the months where these topics have greater average weights</b></i>:

<div style="height: 400px; width: 100%; border: 1px solid #ccc; overflow:scroll; overflow-x: hidden;">
<img src="https://raw.githubusercontent.com/not-even-wong/not-even-wong.github.io/master/_posts/20191117/topics%20clustered%20by%20which%20months%20they%20appear%20in%20copy.jpg">
</div>

I've labelled those topics that I know are related in the syllabus. This is quite a nice result! It looks like topics that surface around the same time of the year each year indeed tend to be topics that should be taught together in the syllabus, so I think it's likely that trends in thread topics are driven by what topic is currently being taught in school for the month.

That's all the analysis I can think of for this dataset. I'd like to fix my scraping algorithm and re-run the data, but it takes several days to scrape everything, during which I can't do any other projects, so I'll put this on the back burner for now. I didn't manage to obtain any relationships affecting thread popularity, but the monthly topic trends are quite interesting. If you've any suggestions, please feel free to contact me using the form below!
