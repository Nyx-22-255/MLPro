.. _RobotHMI:
Robot Manipulator on Homogeneous Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlpro.rl.pool.envs.robotinhtm

.. image:: images/3dmanipulator.png
    :width: 400

This environment represents the robot manipulator in term of mathematical equations.
The mathematical equations are based on rigid body transformation. In this case, the Homogeneous
Transformation Matrix (HTM) is used for the structure. HTM is a matrix that contains both the translation
rotation of a point with respect to some plane.

.. math::

    H=\begin{bmatrix}
    \mathbf{Rot}& \mathbf{Trans}\\ 
    \mathbf{0} & 1
    \end{bmatrix}
    =
    \underbrace{\begin{bmatrix}
    \mathbf{I} & \mathbf{Trans}\\ 
    \mathbf{0} & 1
    \end{bmatrix}}_{translation}
    \underbrace{\begin{bmatrix}
    \mathbf{Rot} & \mathbf{0}\\ 
    \mathbf{0} & 1
    \end{bmatrix}}_{rotation}
    
    
This robotinhtm environment can be imported via:

.. code-block:: python

    import mlpro.rl.pool.envs.robotinhtm


**Prerequisites**

    - `NumPy <https://pypi.org/project/numpy/>`_
    - `PyTorch <https://pypi.org/project/torch/>`_


**General Information**

+------------------------------------+-------------------------------------------------------+
|         Parameter                  |                         Value                         |
+====================================+=======================================================+
| Agents                             | 1                                                     |
+------------------------------------+-------------------------------------------------------+
| Native Source                      | MLPro                                                 |
+------------------------------------+-------------------------------------------------------+
| Action Space Dimension             | [4,]                                                  |
+------------------------------------+-------------------------------------------------------+
| Action Space Base Set              | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| Action Space Boundaries            | [-pi,pi]                                              |
+------------------------------------+-------------------------------------------------------+
| State Space Dimension              | [6,]                                                  |
+------------------------------------+-------------------------------------------------------+
| State Space Base Set               | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| State Space Boundaries             | [-inf,inf]                                            |
+------------------------------------+-------------------------------------------------------+
| Reward Structure                   | Overall reward                                        |
+------------------------------------+-------------------------------------------------------+
  
  
**Action Space**

By default, there are 4 action in this environment. The action space represents the angular velocity of
each joint of the robot manipulator.
  
  
**State Space**

The state space consists of end-effector positions (x,y,z) of the robot manipulator and target positions (x,y,z).
  

**Reward Structure**

By default, the reward structures are shown in the following equation:

.. math::

    reward=-1*\frac{distError}{initDist}-stepReward

    
**Cross Reference**
  + :ref:`Howto RL-ENV-002: SB3 Policy on RobotHTM Environment <Howto Env RL 002>`
  + :ref:`Howto RL-MB-001: MBRL on RobotHTM Environment <Howto MB RL 001>`
  + :ref:`API Reference <target_pool_rl_env_robot_manipulator>`

  
**Citation**

If you apply this environment in your research or work, please :ref:`cite <target_publications>` us and the `original paper <https://ieeexplore.ieee.org/document/10002834>`_.