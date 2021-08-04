#
# StemWF Class
#
# This file is part of CMMLINFLAM
# (https://github.com/I-Bouros/cmml-inflam.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for the forward simulation of the STEM cells
population, both mutated variants and wild types, involved in hematopoesis
in a given tumor.

It uses a Wight-Fisher algorithm.

"""

import numpy as np

from cmmlinflam import StemGillespie, StemGillespieTIMEVAR


class StemWF(StemGillespie):
    r"""StemWF Class:
    Base class for the forward simulation of the evolution of a population
    of STEM cells.

    Three types of cells are considered - those which present mutations that
    give selectional advantage irrespective of environmental conditions (A),
    those which present mutations that give selectional advantage dependent
    on environmental conditions (B) and the wild type cells (WT).

    Cells decay at the same rate independent of their type and devide with
    rates which illsutate their selectional advantage. A wild type cell (WT)
    can mutate to a cell of type A, respectively a cell of type B with constant
    given rates of mutation.

    The system of equations that describe the isolated possible events that can
    occur

    .. math::
        :nowrap:

        \begin{eqnarray}
            A  &\xrightarrow{0} WT \\
            B  &\xrightarrow{0} WT \\
            A  &\xrightarrow{0} B \\
            B  &\xrightarrow{0} A \\
            WT  &\xrightarrow{\mu_{A}} A \\
            WT  &\xrightarrow{\mu_{B}} B
        \end{eqnarray}

    where :math:`\mu_{A}` and :math:`\mu_{B}` are the rate
    of mutation of a WT cell into A cell and respectively, a B cell type. No
    reverse mutations are considered.
    For this class we consider the temporal selectional advatange of the
    B cells always present.

    The total cell population is considered constant so the division of a cell
    is always simultaneous to the death of a cell.

    Therefore, the actual system of equations that describes the model is

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT + WT &\xrightarrow{P_{WT \rightarrow A}} WT + A \\
            WT + WT &\xrightarrow{P_{WT \rightarrow B}} WT + B \\
            A + WT &\xrightarrow{P_{A \rightarrow B}} B + WT \\
            A + WT &\xrightarrow{P_{A \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{B \rightarrow A}} A + WT \\
            A + WT &\xrightarrow{P_{WT \rightarrow B}} A + B \\
            B + WT &\xrightarrow{P_{B \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{WT \rightarrow A}} A + B \\
            A + B &\xrightarrow{P_{A \rightarrow WT}} B + WT \\
            A + B &\xrightarrow{P_{B \rightarrow WT}} A + WT
        \end{eqnarray}

    """
    def __init__(self):
        super().__init__()

    def _prob_WT_sampled(self, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a WT cell in the next generation.

        This event can occur either through the selection of a WT as a parent
        for this daugther cell which will not mutate to either an A or B type
        in the next generation.

        .. math::
            P(\text{WT sampled})=p^{WT}_{k}=\frac{I^{WT}_{k}(\alpha)
            (1-\mu_A-\mu_B)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}
            (\alpha+r\omega_k)}

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        nonmut = (1-mu) * (self.alpha_WT * i_WT) / tot_growth_rate

        return nonmut

    def _prob_A_sampled(self, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a A cell in the next generation.

        This event can occur either through the selection of a A as a parent
        for this daugther cell or the selection of a WT cell which will mutate
        to an A type in the next generation.

        .. math::
            \mathbb{P}(\text{A sampled})=p^{A}_{k}=\frac{I^{A}_{k}(\alpha+s)}
            {I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}(\alpha+r\omega_k)}
            + \frac{\mu_A I^{WT}_{k}(\alpha)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha
            +s)+I^{B}_{k}(\alpha+r\omega_k)}

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        nonmut = (self.alpha_A * i_A) / tot_growth_rate

        # Compute probability of change through mutation
        mutat = (self.mu_A) * (self.alpha_WT * i_WT) / tot_growth_rate

        return (nonmut + mutat)

    def _prob_B_sampled(self, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a B cell in the next generation.

        This event can occur either through the selection of a B as a parent
        for this daugther cell or the selection of a WT cell which will mutate
        to an B type in the next generation.

        .. math::
            \mathbb{P}(\text{B sampled})=p^{B}_{k}=\frac{I^{B}_{k}(\alpha+r)}
            {I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}(\alpha+r\omega_k)}
            + \frac{\mu_B I^{WT}_{k}(\alpha)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha
            +s)+I^{B}_{k}(\alpha+r\omega_k)}

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        nonmut = (self.alpha_B * i_B) / tot_growth_rate

        # Compute probability of change through mutation
        mutat = (self.mu_B) * (self.alpha_WT * i_WT) / tot_growth_rate

        return (nonmut + mutat)

    def one_step_wf(self, i_WT, i_A, i_B):
        """
        Computes one step in the Wright-Fisher algorithm to determine the
        counts of the different species of cells living in the tumor at
        present.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute propensities
        prob_WT = self._prob_WT_sampled(i_WT, i_A, i_B)
        prob_A = self._prob_A_sampled(i_WT, i_A, i_B)
        prob_B = self._prob_B_sampled(i_WT, i_A, i_B)

        probabilities = np.array([prob_WT, prob_A, prob_B], dtype=np.float64)

        # Generate random number for reaction and time to next reaction
        i_WT, i_A, i_B = np.random.multinomial(
            self.N, probabilities/probabilities.sum())

        return (i_WT, i_A, i_B)

    def wf_algorithm_fixed_times(self, start_time, end_time):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        for the given times.

        Parameters
        ----------
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Create timeline vector
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        solution = np.empty((interval, 3), dtype=np.int)
        current_time = start_time
        for t in range(interval):
            i_WT, i_A, i_B = self.one_step_wf(i_WT, i_A, i_B)
            solution[t, :] = np.array([i_WT, i_A, i_B])
            current_time += 1

        return({'state': solution})

    def simulate_fixed_times(self, parameters, start_time, end_time):
        r"""
        Computes the number of each type of cell in a given tumor between the
        given time points.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Check correct format of output
        self._check_times(start_time, end_time)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.wf_algorithm_fixed_times(start_time, end_time)

        output = sol['state']

        return output

    def wf_algorithm_criterion(self, criterion):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        until a criterion is met.

        Parameters
        ----------
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_criterion = 0
        while all(self._evaluate_criterion(criterion, i_WT, i_A, i_B)):
            i_WT, i_A, i_B = self.one_step_wf(i_WT, i_A, i_B)
            time_to_criterion += 1

        return ({
            'steps': time_to_criterion,
            'state': np.array([i_WT, i_A, i_B], dtype=np.int)})

    def simulate_criterion(self, parameters, criterion):
        r"""
        Computes the number of each type of cell in a given tumor until a
        criterion is met.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Check correct format of output
        self._check_criterion(criterion)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.wf_algorithm_criterion(criterion)

        computation_time = sol['steps']
        final_state = sol['state']

        return computation_time, final_state

    def wf_algorithm_fixation(self):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        until fixation.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_fixation = 0
        while self._fixation_condition(i_WT, i_A, i_B):
            i_WT, i_A, i_B = self.one_step_wf(i_WT, i_A, i_B)
            time_to_fixation += 1

        if i_WT == self.N:
            fixed_species = 'WT'
        elif i_A == self.N:
            fixed_species = 'A'
        else:
            fixed_species = 'B'

        return ({
            'steps': time_to_fixation,
            'state': fixed_species})

    def simulate_fixation(self, parameters):
        r"""
        Computes the number of each type of cell in a given tumor until
        fixation.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.

        """
        # Check correct format of output
        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.wf_algorithm_fixation()

        computation_time = sol['steps']
        fixed_species = sol['state']

        return computation_time, fixed_species


class StemWFTIMEVAR(StemGillespieTIMEVAR):
    r"""StemWFTIMEVAR Class:
    Base class for the forward simulation of the evolution of a population
    of STEM cells.

    Three types of cells are considered - those which present mutations that
    give selectional advantage irrespective of environmental conditions (A),
    those which present mutations that give selectional advantage dependent
    on environmental conditions (B) and the wild type cells (WT).

    Cells decay at the same rate independent of their type and devide with
    rates which illsutate their selectional advantage. A wild type cell (WT)
    can mutate to a cell of type A, respectively a cell of type B with constant
    given rates of mutation.

    The system of equations that describe the isolated possible events that can
    occur

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT &\xrightarrow{m} \emptyset \\
            A  &\xrightarrow{m} \emptyset \\
            B  &\xrightarrow{m} \emptyset \\
            \emptyset  &\xrightarrow{\alpha_{WT}} WT \\
            \emptyset  &\xrightarrow{\alpha_{A}} A \\
            \emptyset  &\xrightarrow{\alpha_{B}} B \\
            WT  &\xrightarrow{\mu_{A}} A \\
            WT  &\xrightarrow{\mu_{B}} B
        \end{eqnarray}

    where m is the rate of decay, :math:`\alpha_{WT}`, :math:`\alpha_{A}`,
    and :math:`\alpha_{B}` are the growth rates for the WT, A and B cell
    type respectively and :math:`\mu_{A}` and :math:`\mu_{B}` are the rate
    of mutation of a WT cell into A cell and respectively, a B cell type.
    For this class we consider the temporal selectional advatange of the
    B cells to vary with time.

    The total cell population is considered constant so the division of a cell
    is always simultaneous to the death of a cell.

    Therefore, the actual system of equations that describes the model is

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT + WT &\xrightarrow{P_{WT \rightarrow A}} WT + A \\
            WT + WT &\xrightarrow{P_{WT \rightarrow B}} WT + B \\
            A + WT &\xrightarrow{P_{A \rightarrow B}} B + WT \\
            A + WT &\xrightarrow{P_{A \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{B \rightarrow A}} A + WT \\
            A + WT &\xrightarrow{P_{WT \rightarrow B}} A + B \\
            B + WT &\xrightarrow{P_{B \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{WT \rightarrow A}} A + B \\
            A + B &\xrightarrow{P_{A \rightarrow WT}} B + WT \\
            A + B &\xrightarrow{P_{B \rightarrow WT}} A + WT
        \end{eqnarray}

    """
    def __init__(self):
        super().__init__()

    def _prob_WT_sampled(self, t, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a WT cell in the next generation.

        This event can occur either through the selection of a WT as a parent
        for this daugther cell which will not mutate to either an A or B type
        in the next generation.

        .. math::
            P(\text{WT sampled})=p^{WT}_{k}=\frac{I^{WT}_{k}(\alpha)
            (1-\mu_A-\mu_B)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}
            (\alpha+r\omega_k)}

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        nonmut = (1-mu) * (self.alpha_WT * i_WT) / tot_growth_rate

        return nonmut

    def _prob_A_sampled(self, t, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a A cell in the next generation.

        This event can occur either through the selection of a A as a parent
        for this daugther cell or the selection of a WT cell which will mutate
        to an A type in the next generation.

        .. math::
            \mathbb{P}(\text{A sampled})=p^{A}_{k}=\frac{I^{A}_{k}(\alpha+s)}
            {I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}(\alpha+r\omega_k)}
            + \frac{\mu_A I^{WT}_{k}(\alpha)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha
            +s)+I^{B}_{k}(\alpha+r\omega_k)}

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        nonmut = (self.alpha_A * i_A) / tot_growth_rate

        # Compute probability of change through mutation
        mutat = (self.mu_A) * (self.alpha_WT * i_WT) / tot_growth_rate

        return (nonmut + mutat)

    def _prob_B_sampled(self, t, i_WT, i_A, i_B):
        r"""
        Computes the probability of producing a B cell in the next generation.

        This event can occur either through the selection of a B as a parent
        for this daugther cell or the selection of a WT cell which will mutate
        to an B type in the next generation.

        .. math::
            \mathbb{P}(\text{B sampled})=p^{B}_{k}=\frac{I^{B}_{k}(\alpha+r)}
            {I^{WT}_{k}\alpha+I^{A}_{k}(\alpha+s)+I^{B}_{k}(\alpha+r\omega_k)}
            + \frac{\mu_B I^{WT}_{k}(\alpha)}{I^{WT}_{k}\alpha+I^{A}_{k}(\alpha
            +s)+I^{B}_{k}(\alpha+r\omega_k)}

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute probability of change through non-mutation
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        nonmut = (self.alpha_B * i_B) / tot_growth_rate

        # Compute probability of change through mutation
        mutat = (self.mu_B) * (self.alpha_WT * i_WT) / tot_growth_rate

        return (nonmut + mutat)

    def one_step_wf(self, t, i_WT, i_A, i_B):
        """
        Computes one step in the Wright-Fisher algorithm to determine the
        counts of the different species of cells living in the tumor at
        present.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Compute propensities
        prob_WT = self._prob_WT_sampled(t, i_WT, i_A, i_B)
        prob_A = self._prob_A_sampled(t, i_WT, i_A, i_B)
        prob_B = self._prob_B_sampled(t, i_WT, i_A, i_B)

        probabilities = np.array([prob_WT, prob_A, prob_B], dtype=np.float64)

        # Generate random number for reaction and time to next reaction
        i_WT, i_A, i_B = np.random.multinomial(
            self.N, probabilities/probabilities.sum())

        return (i_WT, i_A, i_B)

    def wf_algorithm_fixed_times(self, start_time, end_time):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        for the given times.

        Parameters
        ----------
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Create timeline vector
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        solution = np.empty((interval, 3), dtype=np.int)
        current_time = start_time
        for t in range(interval):
            i_WT, i_A, i_B = self.one_step_wf(
                current_time, i_WT, i_A, i_B)
            solution[t, :] = np.array([i_WT, i_A, i_B])
            current_time += 1

        return({'state': solution})

    def simulate_fixed_times(
            self, parameters, switch_times, start_time, end_time):
        r"""
        Computes the number of each type of cell in a given tumor between the
        given time points.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Check correct format of output
        self._check_times(start_time, end_time)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.wf_algorithm_fixed_times(start_time, end_time)

        output = sol['state']

        return output

    def wf_algorithm_criterion(self, criterion):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        until a criterion is met.

        Parameters
        ----------
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_criterion = 0
        while all(self._evaluate_criterion(criterion, i_WT, i_A, i_B)):
            i_WT, i_A, i_B = self.one_step_wf(
                time_to_criterion, i_WT, i_A, i_B)
            time_to_criterion += 1

        return ({
            'steps': time_to_criterion,
            'state': np.array([i_WT, i_A, i_B], dtype=np.int)})

    def simulate_criterion(self, parameters, switch_times, criterion):
        r"""
        Computes the number of each type of cell in a given tumor until a
        criterion is met.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Check correct format of output
        self._check_criterion(criterion)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.wf_algorithm_criterion(criterion)

        computation_time = sol['steps']
        final_state = sol['state']

        return computation_time, final_state

    def wf_algorithm_fixation(self):
        """
        Runs the Wright-Fisher algorithm for the STEM cell population
        until fixation.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_fixation = 0
        while self._fixation_condition(i_WT, i_A, i_B):
            i_WT, i_A, i_B = self.one_step_wf(
                time_to_fixation, i_WT, i_A, i_B)
            time_to_fixation += 1

        if i_WT == self.N:
            fixed_species = 'WT'
        elif i_A == self.N:
            fixed_species = 'A'
        else:
            fixed_species = 'B'

        return ({
            'steps': time_to_fixation,
            'state': fixed_species})

    def simulate_fixation(self, parameters, switch_times):
        r"""
        Computes the number of each type of cell in a given tumor until
        fixation.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.

        """
        # Check correct format of output
        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.wf_algorithm_fixation()

        computation_time = sol['steps']
        fixed_species = sol['state']

        return computation_time, fixed_species
