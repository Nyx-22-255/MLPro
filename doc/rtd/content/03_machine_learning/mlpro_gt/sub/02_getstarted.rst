.. _target_getstarted_GT:
Getting Started
---------------

Here is a concise series to introduce all users to the MLPro-GT in a practical way, whether you are a first-timer or an experienced MLPro user.

If you are a first-timer, then you can begin with **Section (1) What is MLPro?**.

If you have understood MLPro but not the game theoretical approach in the engineering field, then you can jump to **Section (2) What is Game Theory?**.

If you have experience in both MLPro and game theory, then you can directly start with **Section (3) What is MLPro-GT?**.

After following the below step-by-step guideline, we expect the user understands the MLPro-GT in practice and starts using MLPro-GT.

**1. What is MLPro?**
   If you are a first-time user of MLPro, you might wonder what is MLPro.
   Therefore, we recommend initially starting with understanding MLPro by checking out the following steps:

   (a) :ref:`MLPro: An Introduction <target_mlpro_introduction>`

   (b) `introduction video of MLPro <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc1.mp4>`_

   (c) :ref:`installing and getting started with MLPro <target_mlpro_getstarted>`

   (d) `MLPro paper in Software Impact journal <https://doi.org/10.1016/j.simpa.2022.100421>`_

**2. What is Game Theory?**
   If you have not dealt with game theory for engineering applications, we recommend starting to understand at least the basic concept of game theory.
   There are plenty of references, articles, papers, books, or videos on the internet that explains the game theory.
   But, for deep understanding, we recommend you to read the book from Dario Bauso, which is `Game Theory with Engineering Applications <https://dl.acm.org/doi/10.5555/2948750>`_.

**3. What is MLPro-GT?**
   We expect that you have a basic knowledge of MLPro and game theory.
   Therefore, you need to understand the overview of MLPro-GT by following the steps below:

   (a) :ref:`MLPro-GT introduction page <target_overview_GT>`

   (b) `Section 5 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_

**4. Understanding Game Board and Player in MLPro-GT**
   First of all, it is important to understand the structure of a game board in MLPro-GT, which can be found on :ref:`this page <target_gb_gt>`.

   In reinforcement learning, we have two types of agents, such as a single-agent RL or a multi-agent RL. Both of the types are covered by MLPro-RL.
   Meanwhile, in MLPro-GT, we focus on a multi-player GT because there are no significant advantages of using game theory for single-player.
   To understand a player in MLPro-GT, you can visit :ref:`this page <target_players_GT>`.

   Then, you can start following some of our howto files and a sample application that shows how to run and train multi-player with their own policy, as follows:

   (a) :ref:`Howto GT-001: Run Multi-Player with Own Policy <Howto GT 001>`

   (b) :ref:`Howto GT-002: Train Multi-Player <Howto GT 002>`

   (c) `Section 6.2 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_

**5. Additional Guidance**
   After following the previous steps, we hope that you could practice MLPro-GT and start using this subpackage for your GT-related activities.
   For more advanced features, we highly recommend you to check out the following howto files:

   (a) :ref:`Howto RL-HT-001: Hyperopt <Howto RL HT 001>`

   (b) :ref:`Howto RL-HT-002: Optuna <Howto RL HT 002>`

   (c) :ref:`Howto RL-ATT-001: Stagnation Detection <Howto RL ATT 001>`
