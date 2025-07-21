from datetime import timedelta
from time import time

import numpy as np
from scipy.sparse import csr_matrix, diags, identity, kron

from models import QBMBase
from simulation import compute_H, compute_rho, get_pauli_kron, get_pauli_kron_hidden
from utils import Discretizer, load_artifact, save_artifact


class sqRBM_em(QBMBase):

    def __init__(
        self,
        prob_data,
        expected_value_V,
        n_visible,
        n_hidden,
        W_init,
        b_init,
        Gamma_init,
        B_freeze=1,
        beta_initial=1.0,
        seed=0,
    ):

        self.prob_data = prob_data
        self.expected_value_V = expected_value_V
        # self.Gamma = Gamma
        self.B_freeze = B_freeze
        self.beta = beta_initial
        super().__init__(n_visible=n_visible, n_hidden=n_hidden, seed=seed, W_init=W_init, b_init=b_init, Gamma_init=Gamma_init)

        self._pauli_kron = get_pauli_kron(self.n_visible, self.n_hidden)
        self._pauli_kron_hidden = get_pauli_kron_hidden(self.n_hidden)

    def train_em(
        self,
        n_epochs,
        n_epochs_m,
        learning_rate,
        epsilon,
        callback=None,
    ):
        
        if not hasattr(self, "callback_history"):
            self.callback_history_epoch = []
            self.callback_history_epoch_m = []

        convergence = True
        for epoch in range(1, n_epochs+1):
            if convergence:
                self.start_time = time()
                self._compute_positive_grads()
                callback_output = callback(self)
                if epoch == 1:
                    self.callback_history_epoch.append(callback_output)
                    self.callback_history_epoch_m.append(callback_output)
                qre_pre = callback_output['qre']
                # kld_pre = callback_output['kld']
                for epoch_m in range(n_epochs_m):
                    self._compute_negative_grads()

                    self.b_copy = self.b.copy()
                    self.W_copy = self.W.copy()
                    self.Gamma_copy = self.Gamma.copy()

                    self._apply_grads_quantum(learning_rate)

                    callback_output = callback(self)
                    qre = callback_output["qre"]
                    kld = callback_output['kld']

                    if (np.abs(qre_pre - qre) > epsilon):
                        # print(f'{epoch}, {epoch_m}, {np.abs(qre_pre - qre)}, {qre}, {kld}')
                        qre_pre = qre
                        # kld_pre = kld
                        self.callback_history_epoch_m.append(callback_output)
                        callback_output_copy = callback_output
                        pass
                    else:
                        self.b = self.b_copy
                        self.W = self.W_copy
                        self.Gamma = self.Gamma_copy
                        if epoch_m == 0:
                            callback_output = callback(self)
                            convergence = False
                        else:
                            callback_output = callback_output_copy
                            self.callback_history_epoch.append(callback_output)
                        break

            else:
                self.callback_history_epoch.append(callback_output)
                self.callback_history_epoch_m.append(callback_output)

        print(f'em_quantum, kld, {kld}')

    def train_gd(
        self,
        n_epochs,
        learning_rate,
        epsilon,
        callback=None,
    ):
        
        if not hasattr(self, "callback_history"):
            self.callback_history = []

        callback_output = callback(self)
        self.callback_history.append(callback_output)
        kld_pre = callback_output['kld']

        convergence = True
        for epoch in range(1, n_epochs+1):
            if convergence:
                self.start_time = time()
                self._compute_positive_grads()
                self._compute_negative_grads()

                self.b_copy = self.b.copy()
                self.W_copy = self.W.copy()
                self.Gamma_copy = self.Gamma.copy()

                self._apply_grads_quantum(learning_rate)

                callback_output = callback(self)
                kld = callback_output['kld']

                if (np.abs(kld - kld_pre) > epsilon):
                    # print(f'{epoch}, {np.abs(kld - kld_pre)}, {kld}')
                    kld_pre = kld
                    self.callback_history.append(callback_output)
                    callback_output_copy = callback_output
                    pass
                else:
                    self.b = self.b_copy
                    self.W = self.W_copy
                    self.Gamma = self.Gamma_copy
                    kld = kld_pre
                    convergence = False
                    pass

            else:
                callback_output = callback_output_copy
                self.callback_history.append(callback_output)

        print(f'gd_quantum, kld, {kld}')

    def _compute_positive_grads(self):
        """
        Computes the gradients for the positive phase
        """

        H_pos_b_expected = np.zeros((1, self.n_hidden))
        H_pos_Gamma_expected = np.zeros((1, self.n_hidden))
        W_pos_expected = np.zeros((self.n_visible, self.n_hidden))
        for i in self.prob_data.keys():
            V_data = np.array(Discretizer.int_to_bit_vector(i, self.n_visible)).reshape(1, -1)
            V_data = self._binary_to_eigen(V_data)

            b_hidden = self.b[self.n_visible :] + V_data @ self.W
            D = np.sqrt(self.Gamma[self.n_visible :]**2 + b_hidden**2)
            H_pos_b = (b_hidden / D) * np.tanh(D)
            H_pos_Gamma = (self.Gamma[self.n_visible :] / D) * np.tanh(D)

            H_pos_b_expected += self.prob_data[i] * H_pos_b
            H_pos_Gamma_expected += self.prob_data[i] * H_pos_Gamma
            W_pos_expected += self.prob_data[i] * (V_data.T @ H_pos_b)

        self.H_pos_b_expected = H_pos_b_expected
        self.H_pos_Gamma_expected = H_pos_Gamma_expected
        self.W_pos_expected = W_pos_expected

        self.grads["b_pos"] = np.concatenate((self.expected_value_V[0], H_pos_b_expected[0]))
        self.grads["Gamma_pos"] = H_pos_Gamma_expected[0]
        self.grads["W_pos"] = W_pos_expected

    def _compute_negative_grads(self):
        """
        Computes the gradients for the negative phase
        """

        p_model, Z = self.compute_p_model(self.prob_data)

        expected_value_model = np.zeros((1, self.n_visible))
        H_neg_b_expected = np.zeros((1, self.n_hidden))
        H_neg_Gamma_expected = np.zeros((1, self.n_hidden))
        W_neg_expected = np.zeros((self.n_visible, self.n_hidden))
        for i in self.prob_data.keys():
            V_data = np.array(Discretizer.int_to_bit_vector(i, self.n_visible)).reshape(1, -1)
            V_data = self._binary_to_eigen(V_data)

            b_hidden = self.b[self.n_visible :] + V_data @ self.W
            D = np.sqrt(self.Gamma[self.n_visible :]**2 + b_hidden**2)
            H_neg_b = (b_hidden / D) * np.tanh(D)
            H_neg_Gamma = (self.Gamma[self.n_visible :] / D) * np.tanh(D)

            expected_value_model += p_model[i] * V_data
            H_neg_b_expected += p_model[i] * H_neg_b
            H_neg_Gamma_expected += p_model[i] * H_neg_Gamma
            W_neg_expected += p_model[i] * (V_data.T @ H_neg_b)

        self.grads["b_neg"] = np.concatenate((expected_value_model[0], H_neg_b_expected[0]))
        self.grads["Gamma_neg"] = H_neg_Gamma_expected[0]
        self.grads["W_neg"] = W_neg_expected

    @property
    def h(self):
        """
        Ising h values. Correspond to b_i = -β * B_freeze * h_i
        """
        return -self.b / (self.beta * self.B_freeze)

    @property
    def J(self):
        """
        Ising J values. Correspond to w_ij = -β * B_freeze * J_ij
        """
        J = np.zeros((self.n_qubits, self.n_qubits))
        J[: self.n_visible, self.n_visible :] = -self.W / (self.beta * self.B_freeze)
        return J
    
    def compute_H_v(self, h, Gamma, i, pauli_kron_hidden):

        V_data = np.array(Discretizer.int_to_bit_vector(i, self.n_visible)).reshape(1, -1)
        V_data = self._binary_to_eigen(V_data)
        b_hidden = self.b[self.n_visible :] + V_data @ self.W
        b_hidden = b_hidden[0]

        H_v_diag = np.zeros(2**self.n_hidden)
        for i in range(0, self.n_hidden):
            # linear terms
            H_v_diag -= b_hidden[i] * pauli_kron_hidden["z_diag", i]

        # return just the diagonal if H is a diagonal matrix
        if np.all(Gamma == 0):
            return np.diag(H_v_diag)

        # off-diagonal terms
        H_v = csr_matrix((2**self.n_hidden, 2**self.n_hidden), dtype=np.float64)
        for i in range(0, self.n_hidden):
            H_v -= Gamma[self.n_visible + i] * pauli_kron_hidden["x", i]

        return (H_v + diags(H_v_diag, format="csr")).toarray()
    
    def compute_rho_V_to_H(self, prob_data):
        rho_V_to_H = {}
        for i in prob_data.keys():
            H_v = self.compute_H_v(self.h, self.Gamma, i, self._pauli_kron_hidden)
            rho_V_to_H[i] = compute_rho(H_v, beta=1)

        return rho_V_to_H
    
    def _bit_to_qubit(self, i):
        n = 0
        a = 0
        V_data = np.array(Discretizer.int_to_bit_vector(i, self.n_visible))
        for i in range(0, self.n_visible):
            x = V_data[i]
            if n==0:
                if x==0:
                    a = np.array([1, 0])
                elif x==1:
                    a = np.array([0, 1])
            else:
                if x==0:
                    b = np.array([1, 0])
                    a = np.kron(a, b)
                elif x==1:
                    b = np.array([0, 1])
                    a = np.kron(a, b)
            n +=1
        return a
    
    def compute_rho_data(self, prob_data):
        rho_data = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        rho_V_to_H = self.compute_rho_V_to_H(prob_data)
        for i in prob_data.keys():
            qubit_x = self._bit_to_qubit(i)
            A = prob_data[i] * np.outer(qubit_x.T, qubit_x)
            B = rho_V_to_H[i]
            C = kron(A, B)
            rho_data += C

        return rho_data

    def compute_p_model(self, prob_data):

        p_model_tilde = np.zeros(2**self.n_visible)
        for i in prob_data.keys():
            V_data = np.array(Discretizer.int_to_bit_vector(i, self.n_visible)).reshape(1, -1)
            V_data = self._binary_to_eigen(V_data)
            b_hidden = self.b[self.n_visible :] + V_data @ self.W

            F = np.exp(np.dot(self.b[: self.n_visible], V_data[0]))
            for j in range(0, self.n_hidden):
                D = np.sqrt((self.Gamma[self.n_visible + j]) ** 2 + b_hidden[0][j]**2)
                F *= np.exp(D) + np.exp(-D)

            p_model_tilde[i] = F

        Z = p_model_tilde.sum()
        p_model = p_model_tilde / Z

        return p_model, Z

    def compute_qre(self, Z):

        qre_1 = (self.expected_value_V[0] * self.b[: self.n_visible]).sum()
        qre_2 = (self.b[self.n_visible :] * self.H_pos_b_expected).sum()
        qre_3 = (self.Gamma[self.n_visible :] * self.H_pos_Gamma_expected).sum()
        qre_4 = (self.W_pos_expected * self.W).sum()

        qre = - (qre_1 + qre_2 + qre_3 + qre_4) + np.log(Z)

        return qre

    def _initialize_weights_and_biases(self, W_init, b_init, Gamma_init):
        """
        Initializes the weights and biases.
        """
        self.W = W_init
        self.b = b_init
        self.Gamma = Gamma_init
