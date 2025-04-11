core.BasePredictor
==================

.. py:module:: core.BasePredictor




Module Contents
---------------

.. py:class:: BasePredictor(model, device, input_path, output_path, make_submission=False, exp_name=None, algo_params=None)

   .. py:attribute:: model


   .. py:attribute:: device


   .. py:attribute:: input_path


   .. py:attribute:: output_path


   .. py:attribute:: make_submission
      :value: False



   .. py:attribute:: exp_name
      :value: None



   .. py:method:: conduct_prediction()


   .. py:method:: write_pred_mask(pred_mask, output_dir, image_name, submission=False)


