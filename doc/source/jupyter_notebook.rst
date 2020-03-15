Remote access to jupyter notebooks
==================================

Jupyter notebooks are extremely convenient to work with in a remote
setting. This short manual explains how to setup a persistent
server running on a **remote machine** (e.g. you machine at work) so that
you can access it from a browser on your **local machine** (e.g. your
laptop at home).

Prerequisites
-------------

We will make use of the following tools:

1. :code:`ssh`: I assume that you have an ssh-client on your laptop that
   allows you to log into your remote machine.
2. :code:`tmux`: `tmux <https://github.com/tmux/tmux/wiki>`_ allows
   running a server in a detached process so that it is not shut down
   when we log out of our ssh session.
3. :code:`jupyter`: Quite obviously, we will make use of `jupyter notebooks <https://jupyter.org/>`_.

Make sure to install these tools if you haven't done so already.

Setting up jupyter
------------------

First, we need to perform some configurations on your **remote machine** to enable
a secure connection to the notebook server. To do so, execute the following steps
on the **remote machine**:

Setup a password
^^^^^^^^^^^^^^^^

Setup a password to access your jupyter notebook server by running the following
on the command line. The command will prompt you to enter a secure password.

.. code-block:: none

  jupyter notebook password

Enable HTTPS
^^^^^^^^^^^^

You should use https to log on to your notebook server to avoid your password
being sent in clear text. Todo so you need to setup a certificate:

.. code-block:: none
   
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ~/.jupyter/mykey.key -out ~/.jupyter/mycert.pem

The command will prompt you to enter a number of descriptive fields. It is sufficient to
just enter your email in the last field and leave the rest blank.

Starting jupyter
----------------

First, open a new tmux window by issuing the :code:`tmux` command from the
command line. You can skip this step and jump directly to the second, but then
your server will not be persistent when you log out from your current ssh
session.


.. code-block:: none

  tmux             


Now start the jupyter notebook server. To be able to connect to the server from your
**remote machine**, we have to tell the server to listen to your public IP. This
is what :code:`--ip=\`hostname -i\`` does. To enable connecting via https, we also
need to provide paths to the certificate and key files we have created.

.. code-block:: none

   jupyter notebook --certfile=~/.jupyter/mycert.pem --keyfile ~/.jupyter/mykey.key --ip=`hostname -i`

The server should now start up and print the IP and port it is listening to.

.. code-block:: none

  [I 16:04:00.182 NotebookApp] The Jupyter Notebook is running at:
  [I 16:04:00.182 NotebookApp] https://<your_computer_name>.rss.chalmers.se:8888/

By default the server will listen to port 8888 but if you have other notebooks
running it will use the next higher one until it finds a free port.
:code:`<your_computer_name>` is the name of your computer that you use to log on
also via ssh. Alternatively you can use the public IP address of your computer.
You can find out the public IP of your computer by running :code:`hostname -i`
from the command line.

You can now detach from the tmux window using the key combination
:code:`CTRL + b` followed by :code:`d`.

.. note::

    After detaching from the :code:`tmux` window, you can list active windows using
    :code:`tmux ls` and then reattach to one of them using :code:`tmux a -t <window_name>`.

Accessing the server
--------------------

You can now connect to your jupyter server from the browser running on your
**local computer** by navigating to :code:`<your_computer_name>.rss.chalmers.se:8888`
in your browser. Your browser will likely print a security warning because we
had to setup the SSL certificate ourselves, but you can safely ignore it.

Accelerating server start-up
----------------------------

To start up a server with a single command, you can combine the above commands
into an alias. To do this add the following to your :code:`~/.bashrc` file:

.. code-block:: none

  alias start_jupyter_server="tmux new-session -d -s jupyter_notebook 'jupyter notebook --certfile=~/.jupyter/mycert.pem --keyfile ~/.jupyter/mykey.key --ip=`hostname -i`'"

Alternative: SSH port forwarding
--------------------------------

As an alternative to starting a server listening on the public IP address of
your computer, you can forward a local port from your **remote machine** via
ssh. For example, if you start a server on your remote machine without the
:code:`--ip` argument, it will listen on :code:`localhost:8888`. You can access
the server by forwarding port :code:`8888` from your **remote machine** to your
local one using an ssh tunnel. The general syntax for ssh port forwarding is:

.. code-block::

  ssh -L <local_port>:localhost:<remote_port> <your_computer_name>.rss.chalmers.se

For example to forward local port :code:`8888` from your **remote  machine** to port
:code:`8888` of your laptop:

.. code-block::

  ssh -L 8888:localhost:8888 <remote_ip>

You can then access the server from your laptop by navigating to
`localhost:8888 localhost:8888` in your browser. You will have to
keep the ssh connection open as long as you want to access the server.
Note also that when your notebook server is listening on another port
than :code:`8888` your will have to adapt the :code:`<remote_port>` argument
accordingly.

