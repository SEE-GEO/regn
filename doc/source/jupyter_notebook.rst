Remote access to jupyter notebooks
==================================

Jupyter notebooks are extremely convenient to work with in a remote
setting. This short manual explains how to setup a persistent
server running on a remote machine (e.g. you machine at work) so that
you can access it from your browser on your local machine (e.g. your
laptop at home).

Prerequisites
-------------

We will make use of the following tools:

1. ssh: I assumed that you have an ssh-client on your laptop that
   allows you to log into your remote machine.
2. `tmux <https://github.com/tmux/tmux/wiki>`_: :code:`tmux` allows
   to run a server in a detached process. This allows us to start
   a server via :code:`ssh` and to keep it running even after we
   have logged back out.
3. jupyter: Quite obviously we will make use of `jupyter notebooks <https://jupyter.org/>`_.

Make sure to install these tools if you haven't done so already.

Setting up jupyter
------------------

First, we need to perform some configurations on your remote machine to ensure
a secure connection to the notebook server. To do so, execute the following steps
on the remote machine:

Setup a password
^^^^^^^^^^^^^^^^

Setup a password to access your jupyter notebook server by running the following
on the command line and entering a secure password.

.. code-block:: bash

jupyter notebook password

Enable HTTPS
^^^^^^^^^^^^

You should use https to log on to your notebook server to avoid your password
being sent in clear text. Todo so you need to setup a certificate:

.. code-block:: bash
   
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ~/.jupyter/mykey.key -out ~/.jupyter/mycert.pem

The command will prompt you to enter a number of descriptive fields, it sufficient if
you just enter your name and email address and leave the rest blank.

Starting jupyter
----------------

First, open a new tmux window by issuing the :code:`tmux` command from the
command line. You can of course skip this step and jump to the second stop, but
then your server will be killed when you log out from your ssh session.


.. code-block:: bash
   tmux             


Now start the jupyter notebook server and tell it to listen on the public IP of your
computer and to use the SSL certificate and key created above.

.. code-block:: bash

   jupyter notebook --certfile=~/.jupyter/mycert.pem --keyfile ~/.jupyter/mykey.key --ip=`hostname -i`

The server should now start up and print the IP and port it is listening to.

.. code-block::

[I 16:04:00.182 NotebookApp] The Jupyter Notebook is running at:
[I 16:04:00.182 NotebookApp] https://<your_computer_name>.rss.chalmers.se:8888/

By default the server will listen to port 8888 but if you have other notebooks
running it will use the next higher one until it finds a free port.
:code:`<your_computer_name>` is the name of your computer that you use to log on
also via ssh. Alternatively you can use the public IP address of your
computer. To find out the public IP of your computer simply run :code:`hostname -i`
from the command line.

You can now detach from the tmux window using the key combination
:code:`CTRL + b` followed by :code:`d`.

Accessing the server
--------------------

You can now connect to your jupyter server from the browser running on your
local computer by navigating :code:`<your_computer_name>.rss.chalmers.se:8888`
in your browser. Your browser will likely print a security warning because we
had to setup the SSL certificate ourselves, but you can safely ignore it.

Accelerating server start-up
----------------------------

To start up a server with a single command, you can combine the above commands
into an alias. To do this add the following to your :code:`~/.bashrc` file:

.. code-block:: bash

alias start_jupyter_server=tmux new-session -d -s jupyter_notebook 'jupyter notebook --certfile=~/.jupyter/mycert.pem --keyfile ~/.jupyter/mykey.key --ip=`hostname -i`'

Alternative: SSH port forwarding
--------------------------------

As an alternative to starting a server listening on the public IP address of
your computer, you can forward a local port from your remote machine via ssh. For
example, if you started a server on your remote machine listening on
:code:`localhost:8889` you can access it by forward port :code:`8888` on your
laptop to the local port on the remote machine:

.. code-block:: bash
ssh -L 8888:localhost:8889 <your_computer_name>.rss.chalmers.se

You can the access the server from your laptop by navigating to
`localhost:8888 localhost:8888` in your browser. Note that you will have to
keep the ssh connection open as long as you want to access the server.
