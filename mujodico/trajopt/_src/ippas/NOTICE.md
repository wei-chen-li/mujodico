# Upstream references

The filter, barrier, line-search, SOC, convergence and primal-regularization
policies follow the local FATROP checkout, specifically
`include/fatrop/ip_algorithm/ip_linesearch.hxx`, `ip_mu_update.hxx`,
`ip_convergence_check.hxx`, `ip_search_dir.hxx`, and
`src/ip_algorithm/ip_filter.cpp`.

FATROP is Copyright (c) Lander Vanroye, KU Leuven. The identified Ipopt-derived
interior-point sources also retain Copyright (C) 2004, 2010 International
Business Machines and others, and are licensed under EPL-2.0. Files containing
adapted policies carry the corresponding SPDX and attribution headers.
The complete upstream notices and licenses are in `licenses/`.

The component unit tests port cases from
`unittest/ip_algorithm/ip_filter_test.cpp`,
`unittest/ip_algorithm/ip_initializer_test.cpp`, and
`unittest/ocp/ip_search_dir_test.cpp`. The point-mass fixture adapts
`unittest/ocp/ocp_test_probem.hpp`. These FATROP test sources fall under the
upstream BSD-2-Clause terms. Test-file headers retain the attribution; the
README documents fixture adaptations and the unsupported terminal equalities.

The existing IPPAS parallel LQR and primal-dual linear-system implementation
is unchanged. This package does not call FATROP or IPOPT at runtime.
