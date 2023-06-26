.. _cwave:

================================
Computation of CWAVE parameters
================================

This section refers to :

.. py:function:: slcl1butils.compute.cwave.compute_cwave_parameters()

CWAVE parmeters are based on the SAR image x-spectrum.

The spectral information of the normalized x-spectrum is decomposed according to orthonormal functions :math:`H_{ij}` defined as tensor products of Gegenbauer polynomial :math:`G_i(\alpha_k(k_x,k_y))` and harmonic :math:`F_j(\alpha_\phi(k_x,k_y))` functions defined in the azimuth :math:`k_y` and range :math:`k_y` wave-number space, following XXX work.

This yields to the general formulation of CWAVE parameters :math:`C_{ij}`:

.. math::
    C_{ij} = \sum_{k_x, k_y}\overline{P}(k_x,k_y) H_{ij}(k_x,k_y)dk_x dk_y,

with :math:`i \in [1,n_k]` and :math:`j \in [1,n_{\phi}]`.
In this study :math:`n_k=4` and :math:`n_{\phi}=5`.

The orthonormal functions are defined such as:

.. math::
    H_{ij}(k_x,k_y) = G_i(\alpha_k) F_j(\alpha_\phi) \eta(k_x , k_y),

where :math:`\eta` writes as:

.. math::
    \eta(k_x,k_y) = \bigg( \frac{2(a_2k_x^2 + 2a_1k_x^4+k_y^2)}{(k_x^2 + k_y^2)(a_2k_x^2 + a_1k_x^4 + k_y^2)(\log k_{\max}-\log k_{\min})}\bigg)^2,
    %\log k_{\max}
    %(\log k_{\max}-\log k_{\min})

with

.. math::
    \begin{align}
      \gamma & =  2 \\
       a_1 & =  \frac{(\gamma^2 - \gamma^4) }{ (\gamma^2 * k_{\min}^2 - k_{\max}^2) }\\
       a_2 & = \frac{ k_{\max}^2 - \gamma^4  k_{\min}^2 }{k_{\max}^2 - \gamma^2 k_{\min}^2}
    \end{align}

In this study, :math:`k_{\min} = 2\pi/600` and :math:`k_{\max} = 2\pi/25` to take benefit of the improved resolution and size of Sentinel-1 SAR images.

:math:`G_i(\alpha_k(k_x,k_y))` writes :

.. math::
    \begin{align}
    G_{i}^{(\lambda)}(x) & = \frac{1}{i} \bigg(2 x (i+\lambda-1) G_{i-1}^{(\lambda)}(x) - (i+2\lambda-2) G_{i-2}^{(\lambda)}(x) \bigg), \textrm{ for } i \ge 2.
    \end{align}

Otherwise:

.. math::
    \begin{align}
    C_{0}^{(\lambda)}(x) & = 1 \\
    C_{1}^{(\lambda)}(x) & = 2 \lambda x
    \end{align}

In this study :math:`\lambda` is set to :math:`3/2`. :math:`F_j(\alpha_\phi(k_x,k_y))` writes :

.. math::
    \begin{align}
    F_j(x) & = \sqrt{2/\pi}\sin\big(jx\big), \textrm{ for n>1, when i is even} \\
    F_j(x) & = \sqrt{2/\pi}\sin\big((j-1)x\big), \textrm{ for n>1, when i is odd}.
    \end{align}

otherwise:

.. math::
    \begin{align}
    F_1(x) & = \sqrt{2/\pi}
    \end{align}

Finally :math:`\alpha_k` and :math:`\alpha_{\phi}` write:

.. math::
    \begin{align}
    \alpha_k & = 2 \frac{ \log\bigg(\sqrt{a_1 k_x^4 + a_2  k_x^2 + k_y^2}\bigg) - \log(k_{\min}) }{ \log(k_{\max})- \log(k_{\min}) } - 1 \\
    \alpha_{\phi} &= \arctan(k_x, k_y).
    \end{align}


