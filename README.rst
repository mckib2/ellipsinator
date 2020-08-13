Ellipsinator
============

Tools for working with ellipses in Python.

Installation
============

Should be an easy pip install:

.. code-block:: bash

    pip install ellipsinator


Usage
=====

To fit an ellipse:

.. code-block:: python

    from ellipsinator import fit_ellipse_halir
    c = fit_ellipse_halir(x, y)

    from ellipsinator import fit_ellipse_fitzgibon
    c = fit_ellipse_fitzgibon(x, y)

    from ellipsinator import fast_guaranteed_ellipse_estimate
    c = fast_guaranteed_ellipse_estimate(x, y)

You can also pass in the measured points as a complex number,
`x + 1j*y`:

.. code-block:: python

    from ellipsinator import fit_ellipse_halir
    c = fit_ellipse_halir(x)

The only function that supports fitting multiple ellipses
simultaneously is `fit_ellipse_halir`:

.. code-block:: python

    assert x.shape == (num_ellipses, num_pts)
    assert y.shape == (num_ellipses, num_pts)
    c = fit_ellipse_halir(x, y)
