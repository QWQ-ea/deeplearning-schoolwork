from pde import Talyer_Green
from AI4XDE.algorithm.PINN import PINN

PDECase = Talyer_Green(
    NumDomain=1000,
    use_VV_form=False,
    loss_weights=None
)
solver=PINN(PDECase)
solver.train()
solver.eval()
solver.save()
PDECase.plot_result(solver)
PDECase.gen_result(solver)