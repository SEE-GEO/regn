Dendrite
========

Dendrite is the name of the network storage that we are using internally to
share data. To access it create a :code:`Dendrite` folder in your home
directory (or any other location you may prefer) and mount it using :code:`sshfs`:

.. code-block:: none

   mkdir ~/Dendrite
   sshfs <your_username>@129.16.35.202:/mnt/array1/share ~/Dendrite

where you have to replace :code:`<your_username>` with your department user name.

It can be useful to define an alias for the above command in your :code:`.bashrc`.
To do so, add the following line to it:

.. code-block:: none

   alias dendrite="sshfs <your_username>@129.16.35.202:/mnt/array1/share ~/Dendrite"


If you need to remount Dendrite, for example after a restart, you can now do so by simply
issuing :code:`mdendrite` from the command line.
