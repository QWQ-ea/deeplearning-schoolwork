import numpy as np
import  deepxde as dde
import deepxde.backend as bkd
from AI4XDE.cases.PDECases import PDECases

data_path = "data_16.h5"
predict_path = "Prediction_512.h5"


class Talyer_Green(PDECases):
    def __init__(
        self,
        t_start=0,
        t_end=30,
        NumDomain=10000,
        exact_PeriodicBC=True,
        use_VV_form=True,
        layer_size=[3] +  [8] * 8  + [3],#3个输入(x,y,t),3个输出(ux,uy,p)
        activation="tanh",#激活函数默认为双曲正切函数
        initializer="Glorot uniform",#默认采用Glorot初始化
        loss_weights=[1, 1, 1, 1000, 1, 1, 1, 1, 1, 1],#输出对应的损失函数的权重
        derivative_order=1,#求导的阶数
        **kwargs,
    ):
        self.t_start = t_start
        self.t_end = t_end
        self.exact_PeriodicBC = exact_PeriodicBC
        self.derivative_order = derivative_order
        self.use_VV_form = use_VV_form
        self.nu = 4.66e-4
        if self.exact_PeriodicBC:
            layer_size[0] = 5
            #loss_weights = loss_weights[:4]
        if self.use_VV_form:
            layer_size[-1] = 2
            loss_weights = loss_weights[:2] + loss_weights[3:]
        super().__init__(
            "Talyer-Green",
            NumDomain,
            layer_size=layer_size,
            activation=activation,
            initializer=initializer,
            loss_weights=loss_weights,
            use_output_transform=use_VV_form,
            #metrics=["l2 relative error"],
            **kwargs,
        )

    def set_loss_weights(self, loss_weights):
        if self.exact_PeriodicBC:
            loss_weights = loss_weights[:4]
        if self.use_VV_form:
            loss_weights = loss_weights[:2] + loss_weights[3:]
        super().set_loss_weights(loss_weights)

    def func_x(self, x):
        return 0.025 * bkd.sin(2 * x[:, 0:1]) * bkd.cos(2 * x[:, 1:2])

    def func_y(self, x):
        return -0.025 * bkd.cos(2 * x[:, 0:1]) * bkd.sin(2 * x[:, 1:2])

    def gen_pde(self):
        def pde_VV(x, y):
            ux = y[:, 0:1]
            uy = y[:, 1:2]
            ro = y[:, 2:3]

            dux_x = dde.grad.jacobian(y, x, i=0, j=0)#i(0,1,2)选择ux,uy,p,j(0,1,2)选择x,y,t,求偏导
            dux_y = dde.grad.jacobian(y, x, i=0, j=1)
            dux_t = dde.grad.jacobian(y, x, i=0, j=2)

            duy_x = dde.grad.jacobian(y, x, i=1, j=0)
            duy_y = dde.grad.jacobian(y, x, i=1, j=1)
            duy_t = dde.grad.jacobian(y, x, i=1, j=2)

            dp_x = dde.grad.jacobian(y, x, i=2, j=0)
            dp_y = dde.grad.jacobian(y, x, i=2, j=1)

            dux_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            dux_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

            duy_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
            duy_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

            pde_x = (
                dux_t
                + ux * dux_x
                + uy * dux_y
                + dp_x
                - self.nu * (dux_xx + dux_yy)
                - self.func_x(x)
            )
            pde_y = (
                duy_t
                + ux * duy_x
                + uy * duy_y
                + dp_y
                - self.nu * (duy_xx + duy_yy)
                - self.func_y(x)
            )
            return [pde_x, pde_y]

        def pde_VP(x, y):
            pde_x, pde_y = pde_VV(x, y)

            dux_x = dde.grad.jacobian(y, x, i=0, j=0)
            duy_y = dde.grad.jacobian(y, x, i=1, j=1)

            pde_z = dux_x + duy_y
            return [pde_x, pde_y, pde_z]

        if self.use_VV_form:
            pde = pde_VV
        else:
            pde = pde_VP

        return pde
#VV形式（Velocity-Vorticity形式）：VV形式将PDE表示为速度（velocity）和涡度（vorticity）的方程组。其中，速度由向量（ux, uy）表示，涡度由标量w表示。这种形式在流体力学中常用，特别适用于描述无粘流体（inviscid flow），如欧拉方程。
#VP形式（Velocity-Pressure形式）：VP形式将PDE表示为速度和压力（pressure）的方程组。其中，速度仍由向量（ux, uy）表示，而压力由标量p表示。这种形式较为普遍，在流体力学中用于描述粘性流体（viscous flow），如纳维-斯托克斯方程（Navier-Stokes equations）。
    def gen_net(self, layer_size, activation, initializer):
        if bkd.backend_name == "paddle":
            net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])#MsFFN类代表一个多尺度前馈神经网络
        else:
            net =dde.nn.FNN(layer_size, activation, initializer)
        if self.use_output_transform:
            net.apply_output_transform(self.output_transform)
        if self.exact_PeriodicBC:
            net.apply_feature_transform(self.input_transform)
        return net

    def output_transform(self, x, y):
        u = dde.grad.jacobian(y, x, i=0, j=1)
        v = -dde.grad.jacobian(y, x, i=0, j=0)
        return bkd.concat((u, v, y[:, 1:2]), axis=1)#对输出进行输出转换

    def input_transform(self, x):
        # omega = 2 * np.pi / ( 2 * np.pi )
        x1_sin = bkd.sin(x[:, 0:1])
        x1_cos = bkd.cos(x[:, 0:1])
        x2_sin = bkd.sin(x[:, 1:2])
        x2_cos = bkd.cos(x[:, 1:2])
        return bkd.concat((x1_sin, x1_cos, x2_sin, x2_cos, x[:, 2:3]), axis=1)#对输入进行特征变换

    def gen_geomtime(self):
        geom = dde.geometry.Rectangle(xmin=[-np.pi, -np.pi], xmax=[np.pi, np.pi])#用于生成几何域
        timedomain = dde.geometry.TimeDomain(self.t_start, self.t_end)#用于生成时间域
        return dde.geometry.GeometryXTime(geom, timedomain)#整合几何域和时间域

    def gen_bc(self):
        X, y = self.get_testdata()
        # when the backend is not pytorch, the following code will cause an error
        # Which can be fixed by changing the error func in PointSetBC class
        observe_y0 = dde.icbc.PointSetBC(X, y, component=[0, 1, 2])#设置观察对象，X为要观察的点的集合，y为要观察的目标值，component为要观察的索引
        bc = [observe_y0]
        return bc
#"exact_PeriodicBC" 代表应用精确的周期边界条件,精确的周期边界条件意味着变量在边界上的值与对应的边界上的值完全匹配，而不引入任何额外的近似
    def gen_data(self):
        bc = self.gen_bc()
        # anchors = self.gen_anchors()
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            bc,
            num_domain=self.NumDomain,#num_domain用于控制空间离散化的程度，而num_boundary用于控制边界离散化的程度
            num_boundary=1000
            # anchors=anchors,#锚点通常会放置在关键时间点上，这些时间点的解是已知的或者可以从其他来源得到的。锚点可以帮助引导求解器更好地逼近未知区域的解。
        )

    def gen_testdata(self):
        import h5py

        X = None
        y = None
        data = h5py.File(data_path, "r")
        for key in data.keys():
            t = float(key)#时间切片值
            if t < self.t_start:
                continue
            if t > self.t_end:
                break
            data_t = np.array(data[key])
            data_t_X = data_t[:, 0:2]
            data_t_X = np.insert(data_t_X, 2, np.ones(data_t_X.shape[0]) * t, axis=1)#在(x,y)数组中插入时间切片值t(x,y,t),data_t_x中不止一个点(x,y)
            data_t_y = data_t[:, 2:]#(ux,uy,p)
            if X is None:
                X = data_t_X
                y = data_t_y
            else:
                X = np.vstack((X, data_t_X))#垂直合并，即行数变，列数不变
                y = np.vstack((y, data_t_y))
        return X, y

    def gen_anchors(self, n=16):
        result = None

        for t in np.linspace(0, 36, 121):
            if t < self.t_start or t > self.t_end:
                continue
            XYT = self.get_domain_at_t(t)
            if result is None:
                result = XYT
            else:
                result = np.vstack((result, XYT))
        return result

    def random_points(self, N, n=16, from_anchors=False):
        if from_anchors:
            X = self.gen_anchors(n)
            X_ids = np.random.choice(
                a=len(X),
                size=N,
                replace=False,
            )
            X_selected = X[X_ids]
        else:
            X_selected = self.geomtime.random_points(N)
        return X_selected

    def get_domain_at_t(self, t):
        d = 2 * np.pi / 512
        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-np.pi, np.pi - d, 512)
                for x2 in np.linspace(-np.pi, np.pi - d, 512)
            ]
        )
        XYT = np.hstack((X, np.full((X.shape[0], 1), t)))#np.full创建了行数与X相同，只有一列为t的数组，np.hstack水平合并X与创建的数组
        return XYT

    def set_axes(self, axes, dim_label):
        label_dim = {
            "x": [-np.pi, np.pi],
            "y": [-np.pi, np.pi],
            "t": [0, self.t_end],
        }
        axes_setlim = [axes.set_xlim, axes.set_ylim]#根据dim_label传参设置轴范围
        axes_setlabel = [axes.set_xlabel, axes.set_ylabel]#根据dim_label传参设置轴标签
        for i, label in enumerate(dim_label):
            axes_setlim[i](*label_dim[label])
            axes_setlabel[i](label)

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes, dim=3)
        axes.scatter(X[:, 0], X[:, 1], X[:, 2])
        return axes

    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes, dim_label=["x", "y"])
        return axes.scatter(X[:, 0], X[:, 1], c=y, s=1, vmin=0, vmax=0.6)
        #c=y表示使用y数组中的值来为每个散点指定颜色。这里假设y是一个与数据集X对应的标签数组。
        #s=1表示散点的大小为1个单位。
        #vmin=0和vmax=0.6表示设置颜色的范围，其中0代表最小值，0.6代表最大值。

    def plot_result(self, solver):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, axes = plt.subplots()
        result_t = []
        if self.t_end == 30:
            linspace_num = 101
        elif self.t_end == 36:
            linspace_num = 121
        for t in np.linspace(self.t_start, self.t_end, linspace_num):
            if t < self.t_start:
                continue
            if t > self.t_end:
                break
            X = self.get_domain_at_t(t)
            model_y = solver.model.predict(X)[:, :2]#预测并取每个预测结果的前两列
            model_y = np.linalg.norm(model_y, axis=1)#求欧几里得距离，axis=1表示以行为单位计算

            axs = self.plot_heatmap_at_axes(X, model_y, axes, title=solver.name)
            result_t.append([axs])

        ani = animation.ArtistAnimation(fig, result_t)#创建一个图像帧的三维对象
        ani.save(
            f"./result/Talyer_Green_{self.t_start}_{self.t_end}.gif", writer="pillow"
        )

    def gen_result_at_t(self, solver, t):
        XYT = self.get_domain_at_t(t)
        predict = solver.model.predict(XYT)
        frame_data = np.concatenate((XYT[:, 0:1], XYT[:, 1:2], predict), axis=1)#拼接数组，axis=1表示同行拼列
        return frame_data

    def gen_result(self, solver, calibrate=False):
        import warnings

        warnings.warn(
            "This function is deprecated and will be removed in future versions, use ResultFileGenerator instead.",
            DeprecationWarning,
        )
        ResultFileGenerator(self, solver).gen_result_file(calibrate=calibrate)


class Separate_Talyer_Green_list:
    def __init__(self, SeparateNum=30, start_time=0, end_time=36, ddt=1, **kwargs):
        self.count = 0
        if not isinstance(ddt, list):
            self.ddt = [ddt] * SeparateNum
        else:
            self.ddt = ddt
        self.start_time = start_time
        self.end_time = end_time
        if isinstance(SeparateNum, list):
            self.SeparateNum = len(SeparateNum) - 1
        else:
            self.SeparateNum = SeparateNum
        self.Separate_Talyer_Green_list = self.gen_Separate_Talyer_Green_list(
            SeparateNum, **kwargs
        )

    def gen_Separate_Talyer_Green_list(self, SeparateNum, **kwargs):
        SeparateNum_list = []
        if isinstance(SeparateNum, list):
            for i in range(len(SeparateNum) - 1):
                SeparateNum_list.append(
                    [SeparateNum[i], SeparateNum[i + 1] + self.ddt[i]]
                )
        else:
            dt = self.end_time / self.SeparateNum
            t = 0
            for i in range(self.SeparateNum):
                SeparateNum_list.append(
                    [
                        t - self.ddt[i],
                        min(t + dt + self.ddt[i], self.end_time),
                    ]
                )
                t += dt

        Separate_Talyer_Green_list = []
        for [t_start, t_end] in SeparateNum_list:
            print(f"{t_start=}, {t_end=}")
            Separate_Talyer_Green_list.append(
                Talyer_Green(t_start=t_start, t_end=t_end, **kwargs)
            )

        return Separate_Talyer_Green_list

    def get_index(self, t):
        index = int(t / self.end_time * self.SeparateNum)
        if index >= self.SeparateNum:
            index = self.SeparateNum - 1
        return index

    def get_domain_at_t(self, t):
        d = 2 * np.pi / 512
        X = np.array(
            [
                [x1, x2]
                for x1 in np.linspace(-np.pi, np.pi - d, 512)
                for x2 in np.linspace(-np.pi, np.pi - d, 512)
            ]
        )
        XYT = np.hstack((X, np.full((X.shape[0], 1), t)))
        return XYT

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < len(self.Separate_Talyer_Green_list):
            result = self.Separate_Talyer_Green_list[self.count]
            self.count += 1
            return result
        else:
            raise StopIteration

    def set_axes(self, axes, dim_label):
        label_dim = {
            "x": [-np.pi, np.pi],
            "y": [-np.pi, np.pi],
            "t": [0, self.end_time],
        }
        axes_setlim = [axes.set_xlim, axes.set_ylim]
        axes_setlabel = [axes.set_xlabel, axes.set_ylabel]
        for i, label in enumerate(dim_label):
            axes_setlim[i](*label_dim[label])
            axes_setlabel[i](label)

    def plot_data(self, X, axes=None):
        from matplotlib import pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        self.set_axes(axes, dim=3)
        axes.scatter(X[:, 0], X[:, 1], X[:, 2])
        return axes

    def plot_heatmap_at_axes(self, X, y, axes, title):
        axes.set_title(title)
        self.set_axes(axes, dim_label=["x", "y"])
        return axes.scatter(X[:, 0], X[:, 1], c=y, s=1, vmin=0, vmax=0.6)

    def plot_result(self, solver_list):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, axes = plt.subplots()
        result_t = []
        if self.end_time == 30:
            linspace_num = 101
        elif self.end_time == 36:
            linspace_num = 121

        for t in np.linspace(0, self.end_time, linspace_num):
            solver = solver_list[self.get_index(t)]
            X = self.get_domain_at_t(t)
            model_y = solver.model.predict(X)[:, :2]
            model_y = np.linalg.norm(model_y, axis=1)

            axs = self.plot_heatmap_at_axes(X, model_y, axes, title=solver.name)
            result_t.append([axs])

        ani = animation.ArtistAnimation(fig, result_t)
        ani.save("./result/Talyer_Green.gif", writer="pillow")

    def gen_result_at_t(self, solver_list, t):
        solver = solver_list[self.get_index(t)]
        XYT = self.get_domain_at_t(t)
        predict = solver.model.predict(XYT)
        frame_data = np.concatenate((XYT[:, 0:1], XYT[:, 1:2], predict), axis=1)
        return frame_data

    def gen_result(self, solver_list, calibrate=False):
        import warnings

        warnings.warn(
            "This function is deprecated and will be removed in future versions, use ResultFileGenerator instead.",
            DeprecationWarning,
        )
        ResultFileGenerator(self, solver_list).gen_result_file(calibrate=calibrate)


class ResultFileGenerator:
    def __init__(self, PDECase, solver):
        self.PDECase = PDECase
        self.solver = solver

    def calibration_rho(self, rho, t):
        rho_mean = np.mean(rho)#计算平均值
        print(f"{t=}, {rho_mean=}")
        if np.abs(rho_mean) > 0.01:
            rho = rho - rho_mean
        return rho

    def frame_data_generator(self, t):
        return self.PDECase.gen_result_at_t(self.solver, t)

    def gen_result_file(self, t_list=np.linspace(0, 30, 101), calibrate=False):
        import h5py

        writeH5File = h5py.File(predict_path, "w")
        for t in t_list:
            frame_data = self.frame_data_generator(t)

            if calibrate:#是否调整
                rho = frame_data[:, -1]#取每行数据的最后一列
                rho = self.calibration_rho(rho, t)
                frame_data[:, -1] = rho

            writeH5File.create_dataset("{:0>5.1f}".format(float(t)), data=frame_data)
        writeH5File.close()


class TwoStageResultFileGenerator(ResultFileGenerator):
    def __init__(self, PDECase_0_30, solver_list, PDECase_30_36, solver_30_36):
        super().__init__(PDECase_0_30, solver_list)
        self.PDECase_30_36 = PDECase_30_36
        self.solver_30_36 = solver_30_36
        self.predict_path = "Prediction_512.h5"

    def frame_data_generator(self, t):
        if t > 30:
            frame_data = self.PDECase_30_36.gen_result_at_t(self.solver_30_36, t)
        else:
            frame_data = self.PDECase.gen_result_at_t(self.solver, t)

        return frame_data
