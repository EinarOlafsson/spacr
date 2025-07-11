train_tools.utils
=================

.. py:module:: train_tools.utils






Module Contents
---------------

.. py:class:: ConfLoader(conf_name)

   Load json config file using DictWithAttributeAccess object_hook.
   ConfLoader(conf_name).opt attribute is the result of loading json config file.


   .. py:class:: DictWithAttributeAccess

      Bases: :py:obj:`dict`


      This inner class makes dict to be accessed same as class attribute.
      For example, you can use opt.key instead of the opt['key'].



   .. py:attribute:: conf_name


   .. py:attribute:: opt


.. py:function:: directory_setter(path='./results', make_dir=False)

   Make dictionary if not exists.


.. py:function:: random_seeder(seed)

   Fix randomness.


.. py:function:: pprint_config(opt)

