Algorithm components
====================

Steps and updates
-----------------

.. autofunction:: OMADS._poll._poll_step.poll_cycle

.. autofunction:: OMADS._poll._poll_step.poll_step

.. autofunction:: OMADS._search._search_step.search_cycle

.. autofunction:: OMADS._search._search_step.search_step

.. autofunction:: OMADS._updates._updates.set_frame_centers_and_hvalues

.. autofunction:: OMADS._updates._updates.update

.. autofunction:: OMADS._metrics._success.compute_success

Preprocessing
-------------

.. autoclass:: OMADS._pre._preprocess.preprocess
   :members: initialize_from_dict

Adaptive barrier
----------------

.. autoclass:: OMADS._barriers._barriers.AdaptiveBarrier
   :members: get_fk, get_ik, get_uk, add_feasible, add_infeasible, update_barrier,
             frame_centers, feasible_frame_center, infeasible_frame_center, extent

Poll directions
---------------

.. autoclass:: OMADS._directions._directions.Dirs2n
   :members: create_housholder, create_poll_set, scale, project_on_mesh_and_snap_to_bounds,
             omit_duplicates

Search samplers
---------------

.. autoclass:: OMADS._heuristics._exploration.EfficientExploration
   :members: generate_sample_points, generate_quad_model_points

.. autoclass:: OMADS._heuristics._exploration.VNS
   :members: run, generate_samples

Meshes
------

.. autoclass:: OMADS._mesh._mesh.Mesh
   :no-members:

.. autoclass:: OMADS._mesh._omesh.Omesh
   :members: enlarge_delta_frame_size, refine_delta_frame_size, project_on_mesh,
             get_delta_mesh_size, get_delta_frame_size

.. autoclass:: OMADS._mesh._gmesh.Gmesh
   :members: enlarge_delta_frame_size, refine_delta_frame_size, project_on_mesh,
             get_delta_mesh_size, get_delta_frame_size

Templates
---------

.. autoclass:: OMADS._templates._optimizer.GenericSamplerBase
   :no-members:
