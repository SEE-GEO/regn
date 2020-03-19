Documentation
=============

Just like science, code is much more useful when it's properly documented.
For this project, we use `Sphinx <https://www.sphinx-doc.org/en/stable/>`_,
which is arguably the most common framework to document Python projects.

Format
------

The documentation consists of source files, which can be found in the
:code:`docs/source` folder. It's structured as a hierarchy of :code:`*.rst`.
The :code:`index.rst` file is the root of the documentation, all other files
that should be included must be linked from this file directly or indirectly.

The documentation in the :code:`*.rst` files can be *built* to create a single
PDF document or a tree of HTML files that can be hosted online.

reStructuredText
----------------

The documentation is written using `reStructuredText
<https://www.sphinx-doc.org/en/stable/usage/restructuredtext/basics.html>`_
markup language. It provides a simple way to add structures to documents
written in plain text. For example you can make a line a section heading
by underlining it with :code:`=`:

.. code-block:: rst

  Section
  =======

To see how it is used it is probably easiest to just have a look a
the :code:`*.rst` file in the :code:`docs` folder our consult the
Sphinx reference on reStructuredText linked above.

Building the documentation
--------------------------

To build the HTML version of the documentation navigate to the :code:`docs`
subfolder and run the following command:

.. code-block:: none

   sphinx-build source .

This will create build the HTML file that make up the documentation hosted
on `simonpf.github.com/regn <https://simonpf.github.com/regn>`_.

.. note ::

   If you add documentation to the project that results in a new HTML file in
   the :code:`docs` folder, you will need to add this to git and push to the
   central repository for it to become available on `simonpf.github.com/regn
   <https://simonpf.github.com/regn>`_.
