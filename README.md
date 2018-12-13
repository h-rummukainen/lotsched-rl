Reinforcement learning experiment with a lot scheduling model
=============================================================

This is research code for the paper "Practical Reinforcement Learning -
Experiences in Lot Scheduling Application" by Hannu Rummukainen and Jukka
K. Nurminen, under review.

Prerequisites
-------------
- OpenAI gym https://gym.openai.com/

- We use a modified copy of deepchem, available at
  https://github.com/h-rummukainen/deepchem

- To get the deepchem dependencies installed, it's probably best to first
  install mainstream deepchem from the Anaconda package distribution
  https://www.anaconda.com/download/

- Perl 5 to post-process run logs

Usage
-----

You can run the script run.sh to repeat the runs in the paper, with random
variations.  See graph.R for a visualization script, which does need manual
editing.
