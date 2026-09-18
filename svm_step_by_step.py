"""
SVM Step-by-Step Solver
=======================
Implements the full mathematical workflow described in the
"SUPPORT VECTOR MACHINE (SVM) — PROFESSIONAL STEP-BY-STEP" prompt:

  1. Identify topic (hard-margin / soft-margin / kernel SVM)
  2. Write given data
  3. Primal problem:  min 1/2||w||^2  s.t. y_i(w^T x_i + b) >= 1
  4. Dual problem via Lagrange multipliers alpha_i
  5. Solve dual QP with KKT conditions
  6. Recover w = sum_i alpha_i y_i x_i   (kernel: implicit)
  7. b = y_i - w^T x_i  from support vectors (averaged)
  8. Margin = 2 / ||w||
  9. Support vectors: alpha_i > 0
 10. Hinge loss  L_i = max(0, 1 - y_i f(x_i))
 11. Prediction: sign(f(x*))  (kernel: sum alpha_i y_i K(x_i, x*) + b)
 12. Verification table of y_i(w^T x_i + b) for every point

Only numpy + scipy are required.
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Kernels (Section 12 of the prompt)
# ----------------------------------------------------------------------
def linear_kernel(x1, x2, **kw):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, gamma=1.0, r=1.0, d=3, **kw):
    return (gamma * np.dot(x1, x2) + r) ** d

def rbf_kernel(x1, x2, gamma=1.0, **kw):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

def sigmoid_kernel(x1, x2, gamma=1.0, r=0.0, **kw):
    return np.tanh(gamma * np.dot(x1, x2) + r)

KERNELS = {
    "linear": linear_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "sigmoid": sigmoid_kernel,
}

# ----------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------
class SVM:
    """
    Solves the SVM dual problem

        max_alpha  sum_i alpha_i
                   - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)

    subject to   sum_i alpha_i y_i = 0
                 0 <= alpha_i        (hard margin)
                 0 <= alpha_i <= C   (soft margin)

    KKT conditions enforced:
      - stationarity      -> w = sum alpha_i y_i x_i
      - dual feasibility  -> alpha_i >= 0 (<= C soft)
      - complementary slackness -> alpha_i > 0 only on margin (or slack)
    """

    def __init__(self, C=None, kernel="linear", **kernel_params):
        """
        C=None       -> hard-margin SVM (Section 3)
        C=float      -> soft-margin SVM (Section 6)
        kernel       -> 'linear' | 'poly' | 'rbf' | 'sigmoid' (Section 12)
        """
        self.C = C
        self.kernel_name = kernel
        self.kernel_fn = KERNELS[kernel]
        self.kernel_params = kernel_params
        self.topic = ("HARD-MARGIN SVM" if C is None
                      else f"SOFT-MARGIN SVM (C={C})")
        if kernel != "linear":
            self.topic += f" with {kernel.upper()} KERNEL"

    # ---------- data ------------------------------------------------
    def fit(self, X, y, tol=1e-6, verbose=True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self.X, self.y = X, y
        n, d = X.shape
        self.n, self.d = n, d

        if verbose:
            self._print_header(X, y, d)

        # ---- Gram matrix K_ij = K(x_i, x_j) -------------------------
        K = np.array([[self.kernel_fn(X[i], X[j], **self.kernel_params)
                       for j in range(n)] for i in range(n)])

        # ---- Dual objective (Section 9) ------------------------------
        # minimize  1/2 alpha^T Q alpha - 1^T alpha
        # Q_ij = y_i y_j K(x_i, x_j)
        Q = np.outer(y, y) * K

        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def grad(alpha):
            return Q @ alpha - np.ones(n)

        # ---- constraints: sum alpha_i y_i = 0 -----------------------
        A_eq = y.reshape(1, -1)
        b_eq = np.zeros(1)

        # ---- bounds: hard margin alpha>=0 ; soft margin 0<=alpha<=C -
        bounds = [(0.0, None)] * n if self.C is None \
                 else [(0.0, self.C)] * n

        alpha0 = np.zeros(n)

        if verbose:
            print("Solving dual QP:  min 1/2 a^T Q a - 1^T a")
            print("  s.t.  sum_i alpha_i y_i = 0,  bounds = "
                  f"{bounds[0]} ...\n")

        res = minimize(objective, alpha0, jac=grad, bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda a: A_eq @ a - b_eq,
                                     "jac": lambda a: A_eq}],
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})

        alpha = np.clip(res.x, 0, None)
        if self.C is not None:
            alpha = np.clip(alpha, 0, self.C)
        self.alpha = alpha
        self.success = res.success

        # ---- support vectors: alpha_i > tol (Section 4) -------------
        self.sv_mask = alpha > tol
        self.support_vectors = X[self.sv_mask]
        self.sv_labels = y[self.sv_mask]
        self.sv_alphas = alpha[self.sv_mask]

        # ---- w = sum alpha_i y_i x_i (stationarity, Section 9) ------
        # Only valid explicitly for the linear kernel.
        self.w = (alpha * y) @ X if self.kernel_name == "linear" else None

        # ---- b from support vectors (Sections 11) --------------------
        self._compute_b(K)

        # ---- margin = 2 / ||w|| (Section 5) --------------------------
        self.margin = (2.0 / np.linalg.norm(self.w)
                       if self.w is not None and np.linalg.norm(self.w) > 0
                       else None)

        if verbose:
            self._print_results()
        return self

    def _compute_b(self, K):
        """b = y_i - sum_j alpha_j y_j K(x_j, x_i), averaged over support
        vectors whose multiplier is not pinned at the upper bound C
        (free support vectors, per complementary slackness)."""
        free = self.sv_mask.copy()
        if self.C is not None:
            free &= self.alpha < self.C - 1e-6
        if not np.any(free):
            free = self.sv_mask
        idx = np.where(free)[0]
        b_vals = np.array([
            self.y[i] - np.sum(self.alpha * self.y * K[:, i]) for i in idx
        ])
        self.b = float(np.mean(b_vals))
        self.b_candidates = b_vals

    # ---------- prediction (Section 13) -------------------------------
    def decision_function(self, Xnew):
        Xnew = np.atleast_2d(np.asarray(Xnew, dtype=float))
        if self.kernel_name == "linear":
            return Xnew @ self.w + self.b
        Ks = np.array([[self.kernel_fn(self.X[i], x, **self.kernel_params)
                        for x in Xnew] for i in range(self.n)])
        return (self.alpha * self.y) @ Ks + self.b

    def predict(self, Xnew):
        return np.sign(self.decision_function(Xnew))

    # ---------- hinge loss (Section 7) --------------------------------
    def hinge_losses(self):
        f = self.decision_function(self.X)
        return np.maximum(0.0, 1.0 - self.y * f)

    def total_hinge_loss(self):
        return float(np.sum(self.hinge_losses()))

    def regularized_objective(self):
        """J = 1/2||w||^2 + C * sum hinge_losses  (linear kernel)."""
        if self.w is None:
            return None
        J = 0.5 * np.dot(self.w, self.w)
        if self.C is not None:
            J += self.C * self.total_hinge_loss()
        return float(J)

    # ---------- verification table (Section 16) ------------------------
    def verification_table(self):
        f = self.decision_function(self.X)
        rows, svs = [], []
        for i in range(self.n):
            yf = self.y[i] * f[i]
            if self.sv_mask[i]:
                status = "SUPPORT VECTOR"
                if self.C is not None and yf < 1 - 1e-6:
                    status += " (on slack)"
            elif yf >= 1:
                status = "correct, not SV"
            else:
                status = "MISCLASSIFIED"
            svs.append(status)
            rows.append((i + 1, self.y[i], f[i], yf, status))
        return rows

    # ---------- pretty printing ----------------------------------------
    def _print_header(self, X, y, d):
        print("=" * 64)
        print(f" TOPIC : {self.topic}")
        print("=" * 64)
        print(f" N (observations) = {self.n},  d (features) = {d}")
        if self.C is not None:
            print(f" C = {self.C}  (soft-margin trade-off parameter)")
        print(f" Kernel = {self.kernel_name}")
        print("\n GIVEN DATA")
        print("-" * 64)
        print(f" {'point':>6} {'y_i':>5}   x_i")
        for i in range(self.n):
            print(f" x_{i+1:<4} {y[i]:>5.0f}   {np.array2string(X[i], precision=3)}")
        print("\n PRIMAL:  min 1/2||w||^2"
              + ("" if self.C is None else f" + C*sum(xi)")
              + "   s.t. y_i(w^T x_i + b) >= 1"
              + ("" if self.C is None else " - xi_i"))
        print(" DUAL  :  max sum a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i,x_j)")
        print("          s.t. sum a_i y_i = 0, "
              + ("a_i >= 0" if self.C is None else "0 <= a_i <= C"))

    def _print_results(self):
        print("\n" + "=" * 64)
        print(" RESULTS")
        print("=" * 64)
        print(f" Dual success        : {self.success}")
        print(f" alpha (multipliers) : {np.round(self.alpha, 4)}")
        print(f" # support vectors   : {int(self.sv_mask.sum())} / {self.n}")
        if self.w is not None:
            print(f" w = sum a_i y_i x_i : {np.round(self.w, 4)}")
            print(f" ||w||               : {np.linalg.norm(self.w):.4f}")
        print(f" b (from free SVs)   : {self.b:.4f}")
        if len(self.b_candidates) > 1:
            print(f"   b candidates      : {np.round(self.b_candidates, 4)}"
                  "  (averaged)")
        if self.margin is not None:
            print(f" MARGIN  = 2/||w||   : {self.margin:.4f}")

        print("\n SUPPORT VECTORS  (alpha_i > 0)")
        print("-" * 64)
        for k in range(len(self.support_vectors)):
            print(f"  a={self.sv_alphas[k]:.4f}  y={self.sv_alphas[k] and self.sv_labels[k]:+.0f}  "
                  f"x={np.array2string(self.support_vectors[k], precision=3)}")

        print("\n VERIFICATION TABLE  (y_i * (w^T x_i + b))")
        print("-" * 64)
        print(f" {'point':>6} {'y_i':>5} {'w^Tx+b':>12} {'y_i(w^Tx+b)':>14}  status")
        for (i, yi, fi, yfi, st) in self.verification_table():
            flag = "" if yfi >= 1 - 1e-6 else "   <-- violates hard-margin constraint"
            print(f" x_{i:<4} {yi:>5.0f} {fi:>12.4f} {yfi:>14.4f}  {st}{flag}")

        hl = self.hinge_losses()
        print("\n HINGE LOSS  L_i = max(0, 1 - y_i f(x_i))")
        print("-" * 64)
        for i in range(self.n):
            print(f" x_{i+1:<4} L_{i+1} = max(0, 1 - ({self.y[i]:+.0f})*({self.decision_function(self.X)[i]:+.4f})) = {hl[i]:.4f}")
        print(f" TOTAL hinge loss = {self.total_hinge_loss():.4f}")
        J = self.regularized_objective()
        if J is not None:
            print(f" J = 1/2||w||^2 + C*sum(L_i) = {J:.4f}")

        if self.w is not None:
            terms = " + ".join(f"({w:.3f})*x{j+1}" for j, w in enumerate(self.w))
            print("\n FINAL HYPERPLANE :  "
                  f"{terms} + ({self.b:.3f}) = 0")
        print(f" CLASSIFIER       :  sign( f(x) ),  "
              f"f(x) = {'w^T x + b' if self.w is not None else 'sum a_i y_i K(x_i,x) + b'}")


# ----------------------------------------------------------------------
# Demo (run: python svm_step_by_step.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":

    print("\n\n############ EXAMPLE 1 : HARD-MARGIN LINEAR SVM ############\n")
    X = np.array([[2, 2], [3, 3], [3, 1],      # class +1
                  [1, 2], [0, 0], [1, 0]])     # class -1
    y = np.array([+1, +1, +1, -1, -1, -1])
    svm = SVM(C=None, kernel="linear").fit(X, y)
    print("\n Predictions on training data:",
          svm.predict(X).astype(int))

    print("\n\n############ EXAMPLE 2 : SOFT-MARGIN SVM, C = 1 ############\n")
    X2 = np.array([[2, 2], [3, 3], [2.5, 0.5],
                   [1, 2], [0, 0], [1.8, 1.8]])   # last point is an outlier
    y2 = np.array([+1, +1, +1, -1, -1, -1])
    svm2 = SVM(C=1.0, kernel="linear").fit(X2, y2)
    print("\n Predictions:", svm2.predict(X2).astype(int))

    print("\n\n############ EXAMPLE 3 : RBF KERNEL SVM ############\n")
    # XOR dataset - not linearly separable, needs a kernel
    X3 = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y3 = np.array([-1, +1, +1, -1])
    svm3 = SVM(C=10.0, kernel="rbf", gamma=1.0).fit(X3, y3)
    print("\n Predictions (XOR):", svm3.predict(X3).astype(int))
    print(" True labels       :", y3)