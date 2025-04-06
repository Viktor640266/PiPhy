import numpy as np
import matplotlib.pyplot as plt

class Kinematics():
    # В этой функции есть аргумент planet. 
    # planet это космический объект для которого в классе установится ускорение свободного падения, которое учавствует в расчёте в некоторых из функций
    # Выбирайте planet из словаря g_dict
    def __init__(self, planet: str):
        self.g_dict = {"Mercury": 3.7, "Venus": 8.87, "Earth": 9.81, "Mars": 3.71, "Jupiter": 24.79, 
                       "Saturn": 10.44, "Uranus": 8.69, "Neptune": 11.15, "Pluto": 0.62, "Sun": 274.0,
                       "Io": 1.79, "Moon": 1.62, "Europa": 1.31, "Ganymede": 1.43, "Callisto": 1.24, 
                       "Enceladus": 0.113, "Titan": 1.35, "Rhea": 0.264, "Miranda": 0.079, "Ariel": 0.269,
                       "Triton": 0.779, "Charon": 0.288, "Amalthea": 0.020, "Himalia": 0.062, 
                       "Mimas": 0.064, "Tethys": 0.147, "Dione": 0.233, "Hyperion": 0.017, "Iapetus": 0.224,
                       "Phoebe": 0.042, "Umbriel": 0.2, "Titania": 0.38, "Oberon": 0.35, "Nereid": 0.003,
                       "Styx": 0.005, "Nix": 0.012, "Kerberos": 0.003, "Hydra": 0.008}
        for key, value in self.g_dict.items():
            if planet == key:
                self.g = value

    # Возвращает вектор импульса и кинетическую энергию для каждого из введённых тел
    def impulse_k_energy(self, m: list, v0_vectors: list):
        impulse_list = []
        k_energy_list = []
        for i in range(len(m)):
            impulse = [v0_vectors[i][0] * m[i], v0_vectors[i][1] * m[i], v0_vectors[i][2] * m[i]]
            k_energy = m[i] * np.linalg.norm(v0_vectors[i]) ** 2 / 2

            impulse_list.append(impulse)
            k_energy_list.append(k_energy)

        return impulse_list, k_energy_list

    # Возвращает вектор импульса и кинетическую энергию всей системы тел
    def all_impulse_k_energy(self, impulse_list: list, k_energy_list: list):
        system_impulse = []
        for k in range(len(impulse_list) - 1):
            for l in range(len(impulse_list[k])):
                system_impulse.append(impulse_list[k][l] + impulse_list[k + 1][l])
        system_k_energy = np.sum(k_energy_list)

        return system_impulse, system_k_energy
    
    # Возвращает координаты центра масс всей системы
    def mass_center(self, mass_list: list, coords_list: list, velocity_list: list, coords_dimension: int):
        coords = []
        velocity = []

        for i in range(len(coords_list)):
            for j in range(coords_dimension):
                coords.append(mass_list[i] * coords_list[i][j])
                velocity.append(mass_list[i] * velocity_list[i][j])

        x_list_c = []
        y_list_c = []
        x_list_v = []
        y_list_v = []

        for k in range(len(coords)):
            if k % 2 == 0:
                x_list_c.append(coords[k])
                x_list_v.append(velocity[k])
            else:
                y_list_c.append(coords[k])
                y_list_v.append(velocity[k])
        x_c = np.sum(x_list_c)
        y_c = np.sum(y_list_c)
        x_v = np.sum(x_list_v)
        y_v = np.sum(y_list_v)

        m_c_coords = [(1 / np.sum(mass_list)) * x_c, (1 / np.sum(mass_list)) * y_c]
        m_c_velocity = [(1 / np.sum(mass_list)) * x_v, (1 / np.sum(mass_list)) * y_v]
    
        return m_c_coords, m_c_velocity

    # Поворот координат в 2D измерении
    def rotate_coords_2D(self, coords_list: list, angle: float):
        if len(coords_list) == 2:
            x1 = coords_list[0] * np.cos(angle) - coords_list[1] * np.sin(angle)
            y1 = coords_list[0] * np.sin(angle) + coords_list[1] * np.cos(angle)

            return [x1, y1]
    
    # Поворот координат в 3D измерении - в rotate_axes нужно ввести плоскость вокруг которой хотите повернуть координаты
    def rotate_coords_3D(self, coords_list: list, angle: float, rotate_axes: list):
        if len(coords_list) == 3:
            axes_dict = {"YZ": np.array([[1, 0, 0], 
                                        [0, np.cos(angle), -np.sin(angle)],
                                        [0, np.sin(angle), np.cos(angle)]]),
                        "XZ": np.array([[np.cos(angle), 0, np.sin(angle)], 
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]]),
                        "XY": np.array([[np.cos(angle), -np.sin(angle), 0],  
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]]),
                        "yz": np.array([[1, 0, 0], 
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]]),
                        "xz": np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]]),
                        "xy": np.array([[np.cos(angle), -np.sin(angle), 0], 
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])}
            
            rotate_axes = rotate_axes.split()
            rotate_axes = np.array(rotate_axes)

            coords = np.array(coords_list).reshape(3, 1)

            if len(rotate_axes) == 1:
                xyz = axes_dict[rotate_axes[0]] @ coords
            if len(rotate_axes) == 2:
                xyz = axes_dict[rotate_axes[0]] @ axes_dict[rotate_axes[1]] @ coords
            if len(rotate_axes) == 3:
                xyz = axes_dict[rotate_axes[0]] @ axes_dict[rotate_axes[1]] @ axes_dict[rotate_axes[2]] @ coords

            return [xyz[0][0], xyz[1][0], xyz[2][0]]

    # Поворот координат в ND измерении - в rotate axes нужно написать список осей из которых состоит плоскость поворота
    # Например, ["x2", "x5"]. Да, здесь все оси обозначаются "x{номер оси}".
    def rotate_coords_nD(self, coords_list: list, angle: float, rotate_axes: list):
        coords_dict = {}
        ij_list = []
        rotate_matrix = np.zeros((len(coords_list), len(coords_list)))
        for i in range(len(coords_list)):
            coords_dict[f"x{i + 1}"] = coords_list[i]
            coords_dict[f"X{i + 1}"] = coords_list[i]
            if f"x{i + 1}" == rotate_axes[0] or f"x{i + 1}"  == rotate_axes[1] or f"X{i + 1}" == rotate_axes[0] or f"X{i + 1}"  == rotate_axes[1]:
                ij_list.append(i + 1)
        for a in range(np.shape(rotate_matrix)[0]):
            for b in range(np.shape(rotate_matrix)[1]):
                if a == ij_list[0] and b == ij_list[1]:
                    rotate_matrix[a - 1][b - 1] = -np.sin(angle)
                if a == ij_list[1] and b == ij_list[0]:
                    rotate_matrix[a - 1][b - 1] = np.sin(angle)
                if a == ij_list[0] and b == ij_list[0]:
                    rotate_matrix[a - 1][b - 1] = np.cos(angle)
                if a == ij_list[1] and b == ij_list[1]:
                    rotate_matrix[a - 1][b - 1] = np.cos(angle)
                else:
                    for k in range(a):
                        rotate_matrix[a][a] = 1

        rotate_matrix = np.array(rotate_matrix)
        coords_list = np.array(coords_list)

        xyz = rotate_matrix @ coords_list

        return xyz
    
    # С 152 по 204 строку функции для вычисления неизвестных параметров параболической траектории через другие параметры, которые известны
    def angle__v0x(self, v0, v0x):
        return np.acos(v0x / v0)
    def angle__v0y(self, v0, v0y):
        return np.asin(v0y / v0)
    
    def v0__v0x_angle(self, v0x, angle):
        return v0x / np.cos(angle)
    def v0__v0y_angle(self, v0y, angle):
        return v0y / np.sin(angle)
    def v0__L_angle(self, L, angle):
        return np.sqrt(L / np.sin(2 * angle) * self.g)
    
    def v0x__L_tfall(self, L, tfall):
        return(L / tfall)
    def v0x__xt_tfall(self, xt, tfall):
        return xt / tfall
    def v0x__v0_angle(self, v0, angle):
        return v0 * np.cos(angle)

    def v0y__tfall(self, tfall):
        return tfall * self.g / 2
    def v0y__H(self, H):
        return H * (2 * self.g)
    def v0y__vy_tfall(self, vy, tfall):
        return vy + (self.g * tfall)
    def v0y__yt_tfall(self, yt, tfall):
        return yt + (self.g * tfall ** 2) / tfall
    def v0y__v0_angle(self, v0, angle):
        return v0 * np.sin(angle)
    
    def H(self, v0y):
        return v0y ** 2 / (2 * self.g)
    
    def tfall__v0y(self, v0y):
        return 2 * v0y / self.g
    def tfall__L_v0x(self, L, v0x):
        return L / v0x

    def L__v0x_tfall(self, v0x, tfall):
        return v0x * tfall
    def L__v0_angle(self, v0, angle):
        return v0 ** 2 / self.g * np.sin(2 * angle)
    
    def xt(self, v0x, tfall):
        return v0x * tfall
    def yt(self, v0y, tfall):
        return v0y * tfall - self.g * tfall ** 2 / 2

    def vx(self, v0x):
        return v0x
    def vy(self, v0y, tfall):
        return v0y - (self.g * tfall)
    
    # Это функция для вычисления траектории тела через два известных параметра траектории
    # Например, через угол и начальную скорость - angle и v0
    # В аргументах нужно написать название параметра=значение параметра
    # L, l - длина траектории 
    # Angle, angle - угол под которым запускается тело
    # v0x, V0X - x-компонента вектора начальной скорости v0
    # v0y, V0Y - y-компонента вектора начальной скорости v0
    # Tfall, tfall, T, t - время падения тела
    # H, h - максимальная высота которой достигает тело
    def trajectory_2_parameters(self, **kwargs):
        x_list = []
        y_list = []
        keys = list(kwargs.keys())
        for i in range(len(keys) - 1):
            if (keys[i] == "v0" or keys[i] == "V0") and (keys[i + 1] == "angle" or keys[i + 1] == "Angle"):
                v0 = kwargs[keys[i]]
                angle = kwargs[keys[i + 1]]
                v0x = self.v0x__v0_angle(v0, angle)
                v0y = self.v0y__v0_angle(v0, angle)
                tfall = abs(self.tfall__v0y(v0y))
            elif (keys[i] == "angle" or keys[i] == "Angle") and (keys[i + 1] == "v0" or keys[i + 1] == "V0"):
                v0 = kwargs[keys[i + 1]]
                angle = kwargs[keys[i]]
                v0x = self.v0x__v0_angle(v0, angle)
                v0y = self.v0y__v0_angle(v0, angle)
                tfall = abs(self.tfall__v0y(v0y))

            elif (keys[i] == "angle" or keys[i] == "Angle") and (keys[i + 1] == "tfall" or keys[i + 1] == "Tfall"):
                angle = kwargs[keys[i]]
                tfall = kwargs[keys[i + 1]]
                v0y = self.v0y__tfall(tfall)
                v0 = self.v0__v0y_angle(v0y, angle)
                v0x = self.v0x__v0_angle(v0, angle)
            elif (keys[i] == "tfall" or keys[i] == "Tfall") and (keys[i + 1] == "angle" or keys[i + 1] == "Angle"):
                angle = kwargs[keys[i + 1]]
                tfall = kwargs[keys[i]]
                v0y = self.v0y__tfall(tfall)
                v0 = self.v0__v0y_angle(v0y, angle)
                v0x = self.v0x__v0_angle(v0, angle)

            elif (keys[i] == "L" or keys[i] == "l") and (keys[i + 1] == "tfall" or keys[i + 1] == "Tfall" or keys[i + 1] == "T" or keys[i + 1] == "t"):
                l = kwargs[keys[i]]
                tfall = kwargs[keys[i + 1]]
                v0x = self.v0x__L_tfall(l, tfall)
                v0y = self.v0y__tfall(tfall)
            elif (keys[i] == "tfall" or keys[i] == "Tfall" or keys[i] == "T" or keys[i] == "t") and (keys[i + 1] == "L" or keys[i + 1] == "l"):
                l = kwargs[keys[i + 1]]
                tfall = kwargs[keys[i]]
                v0x = self.v0x__L_tfall(l, tfall)
                v0y = self.v0y__tfall(tfall)

            elif (keys[i] == "L" or keys[i] == "l") and (keys[i + 1] == "H" or keys[i + 1] == "h"):
                l = kwargs[keys[i]]
                h = kwargs[keys[i + 1]]
                v0y = self.v0y__H(h)
                tfall = self.tfall__v0y(v0y)
                v0x = self.v0x__L_tfall(l, tfall)
            elif (keys[i] == "H" or keys[i] == "h") and (keys[i + 1] == "L" or keys[i + 1] == "l"):
                l = kwargs[keys[i + 1]]
                h = kwargs[keys[i]]
                v0y = self.v0y__H(h)
                tfall = self.tfall__v0y(v0y)
                v0x = self.v0x__L_tfall(l, tfall)

            elif (keys[i] == "v0x" or keys[i] == "V0X") and (keys[i + 1] == "tfall" or keys[i + 1] == "Tfall" or keys[i + 1] == "T" or keys[i + 1] == "t"):
                v0x = kwargs[keys[i]]
                tfall = kwargs[keys[i + 1]]
                v0y = self.v0y__tfall()
            elif (keys[i] == "tfall" or keys[i] == "Tfall" or keys[i] == "T" or keys[i] == "t") and (keys[i + 1] == "v0x" or keys[i + 1] ==  "V0X"):
                v0x = kwargs[keys[i]]
                tfall = kwargs[keys[i + 1]]
                v0y = self.v0y__tfall(tfall)

            elif (keys[i] == "v0x" or keys[i] == "V0X") and (keys[i + 1] == "H" or keys[i + 1] == "h"):
                v0x = kwargs[keys[i]]
                h = kwargs[keys[i + 1]]
                v0y = self.v0y__H(h)
                tfall = self.tfall__v0y(v0y)
            elif (keys[i] == "H" or keys[i] == "h") and (keys[i + 1] == "v0x" or keys[i + 1] == "V0X"):
                h = kwargs[keys[i]]
                v0x = kwargs[keys[i + 1]]
                v0y = self.v0y__H(h)
                tfall = self.tfall__v0y(v0y)

            elif (keys[i] == "v0" or keys[i] == "V0") and (keys[i + 1] == "H" or keys[i + 1] == "h"):
                v0 = kwargs[keys[i]]
                h = kwargs[keys[i + 1]]
                v0y = self.v0y__H(h)
                angle = self.angle__v0y(v0, v0y)
                v0x = self.v0x__v0_angle(v0, angle)
            elif (keys[i] == "H" or keys[i] == "h") and (keys[i + 1] == "v0" or keys[i + 1] == "V0"):
                v0 = kwargs[keys[i]]
                h = kwargs[keys[i + 1]]
                v0y = self.v0y__H(h)
                angle = self.angle__v0y(v0, v0y)
                v0x = self.v0x__v0_angle(v0, angle)
                
            elif (keys[i] == "v0x" or keys[i] == "V0X") and (keys[i + 1] == "v0y" or keys[i + 1] == "V0Y"):
                v0x = kwargs[keys[i]]
                v0y = kwargs[keys[i + 1]]
                tfall = self.tfall__v0y(v0y)
            elif keys[i] == "v0y" and keys[i + 1] == "v0x" or keys[i] == "V0Y" and keys[i + 1] == "V0X":
                v0x = kwargs[keys[i + 1]]
                v0y = kwargs[keys[i]]
                tfall = self.tfall__v0y(v0y)
                
        if tfall < 2:
            tfall *= 10

        for i in range(abs(int(tfall))):
            xt = self.xt(v0x, i)
            yt = self.yt(v0y, i)
            x_list.append(xt)
            y_list.append(yt)

        return x_list, y_list
    
    # Возвращает координаты необходимые для расчёта траектории маятника и фазовой траектории маятника
    # Маятник - диск радиуса disk_radius вращающийся вокруг оси которая находится в центре окружности
    def pendulum_disk_rotating_its_axis(self, m: float, v0: float, disk_radius: float):
        x_list = []
        y_list = []

        angle_list = []
        angle_v_list = []

        i = (1/2) * m * disk_radius ** 2

        t = 2 * np.pi * np.sqrt(i / (m * self.g * disk_radius))
        t_linspace = np.linspace(0, t, 100)
        t_linspace_2 = np.linspace(0, 2 * t, 100)
        f = np.sqrt(self.g / disk_radius)

        angle0 = v0 / f

        for j in t_linspace:
            angle = angle0 * np.cos(f * j)
            x = disk_radius * np.sin(angle)
            y = -disk_radius * np.cos(angle)
            x_list.append(x)
            y_list.append(y)

        for k in t_linspace_2:
            angle = angle0 * np.cos(f * k)
            angle_v = -angle0 * f * np.sin(f * k)
            angle_list.append(angle)
            angle_v_list.append(angle_v)

        return x_list, y_list, angle_list, angle_v_list
    
    # Возвращает то же самое что и в прошлой функции
    # Маятник - стержень длины rod_length подвешенный и вращающийся за один конец вокруг оси которая находится на конце
    def pendulum_rod_rotating_one_end(self, m: float, v0: float, rod_length: float):
        x_list = []
        y_list = []

        angle_list = []
        angle_v_list = []

        i = (1/3) * m * rod_length ** 2

        t = 2 * np.pi * np.sqrt(i / (m * self.g * rod_length))
        t_linspace = np.linspace(0, t, 100)
        t_linspace_2 = np.linspace(0, 2 * t, 100)
        f = np.sqrt(self.g / rod_length)

        angle0 = v0 / f

        for j in t_linspace:
            angle = angle0 * np.cos(f * j)
            x = rod_length * np.sin(angle)
            y = -rod_length * np.cos(angle)
            x_list.append(x)
            y_list.append(y)

        for k in t_linspace_2:
            angle = angle0 * np.cos(f * k)
            angle_v = -angle0 * f * np.sin(f * k)
            angle_list.append(angle)
            angle_v_list.append(angle_v)

        return x_list, y_list, angle_list, angle_v_list

    # Возвращает то же самое что и две прошлые функции
    # Маятник - диск радиуса disk_radius, прикрепленный за ребро (бок) и вращающийся вокруг оси которая находится на ребре (на боку) окружности
    def pendulum_disk_rotating_axis_at_its_edge(self, m: float, v0: float, disk_radius: float):
        x_list = []
        y_list = []

        angle_list = []
        angle_v_list = []
        
        i = (3/2) * m * disk_radius ** 2

        t = 2 * np.pi * np.sqrt(i / (m * self.g * disk_radius))
        t_linspace = np.linspace(0, t, 100)
        t_linspace_2 = np.linspace(0, 2 * t, 100)
        f = np.sqrt(self.g / disk_radius)

        angle0 = v0 / f

        for j in t_linspace:
            angle = angle0 * np.cos(f * j)
            x = disk_radius * np.sin(angle)
            y = -disk_radius * np.cos(angle)
            x_list.append(x)
            y_list.append(y)

        for k in t_linspace_2:
            angle = angle0 * np.cos(f * k)
            angle_v = -angle0 * f * np.sin(f * k)
            angle_list.append(angle)
            angle_v_list.append(angle_v)

        return x_list, y_list, angle_list, angle_v_list
    
    # Возвращает импульс тела
    def impulse(self, m: float, v: float):
        return m * v

    # Возвращает скорости и импульсы двух тел после их упругого столкновения
    def collisions_elastic(self, m_list: list, v_list: list):
        m1 = m_list[0]
        m2 = m_list[1]

        v1 = v_list[0]
        v2 = v_list[1]

        v1_after = []
        v2_after = []
        p1 = []
        p2 = []
        p1_after = []
        p2_after = []

        for i in range(len(v1)):
            v1_after.append(((m1 - m2) * v1[i] + 2 * m2 * v2[i]) / (m1 + m2))
            v2_after.append(((m2 - m1) * v2[i] + 2 * m1 * v1[i]) / (m1 + m2))
            p1.append(self.impulse(m1, v1[i]))
            p2.append(self.impulse(m2, v2[i]))
        for j in range(len(v1_after)):
            p1_after.append(m1 * v1_after[j])
            p2_after.append(m2 * v2_after[j])

        return v1_after, p1_after, v2_after, p2_after

    # Возвращает скорости и импульсы двух тел после их неупругого столкновения
    # 0 < coefficient_of_restitution < 1
    def collisions_not_elastic(self, m_list: list, v_list: list, coefficient_of_restitution: float):
        v1_after = []
        v2_after = []
        p1 = []
        p2 = []
        p1_after = []
        p2_after = []

        e = coefficient_of_restitution

        m1 = m_list[0]
        m2 = m_list[1]

        v1 = v_list[0]
        v2 = v_list[1]
        
        for i in range(len(v1)):
            v1_after.append((m1 * v1[i] + m2 * v2[i] - m2 * e * (v1[i] - v2[i])) / (m1 + m2))
            v2_after.append((m1 * v1[i] + m2 * v2[i] + m1 * e * (v1[i] - v2[i])) / (m1 + m2))
            p1.append(self.impulse(m1, v1[i]))
            p2.append(self.impulse(m1, v2[i]))
        for j in range(len(v1_after)):
            p1_after.append(self.impulse(m1, v1_after[j]))
            p2_after.append(self.impulse(m1, v2_after[j]))

        return v1_after, p1_after, v2_after, p2_after

    # Возвращает скорость и импульс тела которое получилось в результате абсолютно неупругого столкновения двух тел (тела слиплись)
    def collisions_absolute_not_elastic(self, m_list: list, v_list: list):
        m1 = m_list[0]
        m2 = m_list[1]

        v1 = v_list[0]
        v2 = v_list[1]

        p1 = []
        p2 = []

        v_after = []
        m_after = m1 + m2

        p_after = []

        for i in range(len(v1)):
            v_after.append((m1 * v1[i] + m2 * v2[i]) / (m1 + m2))
            p1.append(self.impulse(m1, v1[i]))
            p2.append(self.impulse(m2, v2[i]))
        for j in range(len(v_after)):
            p_after.append(self.impulse(m_after, v_after[j]))

        return m_after, v_after, p_after
    
    # Рисует график параболической траектории через функцию trajectory_2_parameters
    # func - это "self.trajectory_2_parameters(параметр_1, параметр_2)"
    def plot_trajectory(self, func):
        x_list, y_list = eval(func)

        plt.plot(x_list, y_list)
        plt.title("Trajectory")
        plt.grid()
        plt.show()

        plt.plot(x_list, y_list)
        plt.axis("equal")
        plt.title("Trajectory With Equal Axes")
        plt.grid()
        plt.show()

    # Рисует график маятника через одну из трёх функций для маятников
    # func - "self.pendulum_disk_rotating_its_axis(параметр1, параметр2, параметр3)" или
    # "self.pendulum_rod_rotating_one_end(параметр1, параметр2, параметр3)" или
    # "self.pendulum_disk_rotating_its_axis_at_its_edge(параметр1, параметр2, параметр3)"
    def plot_pendulum(self, func):
        x_list, y_list, angle_list, angle_v_list = eval(func)

        plt.plot(x_list, y_list)
        plt.title("Pendulum Trajectory")
        plt.grid()
        plt.show()

        plt.plot(angle_list, angle_v_list)
        plt.title("Phase Trajectory")
        plt.grid()
        plt.show()

        plt.plot(angle_list, angle_v_list)
        plt.axis("equal")
        plt.title("Phase Trajectory With Equal Axes")
        plt.grid()
        plt.show()

    

s = Kinematics("Sun")
e = Kinematics("Earth")
s.trajectory_2_parameters(tfall=10, L=100)
s.impulse_k_energy([0.2, 0.5], [[3, 2, 0], [3, 2, 0]])
s.all_impulse_k_energy([[0.6000000000000001, 0.4, 0.0], [1.5, 1.0, 0.0]], [(1.2999999999999998), (3.2499999999999996)])
s.mass_center([5, 2], [[0, 4], [3, 5]], [[2, 5], [2, 5]], 2)
s.rotate_coords_2D([2, 5], 45)
s.rotate_coords_3D([2, 5, 7], 45, "XY")
s.rotate_coords_nD([2, 5, 7, 20, 13], 45, ["x4", "x1"]) 
s.pendulum_disk_rotating_axis_at_its_edge(100, 100, 10)
s.pendulum_disk_rotating_its_axis(100, 100, 10)
s.pendulum_rod_rotating_one_end(100, 100, 10)
s.collisions_elastic([2, 5], [[2, 8], [9, 1]])
s.collisions_not_elastic([2, 5], [[2, 8], [9, 1]], 0.3)
s.collisions_absolute_not_elastic([2, 5], [[2, 8], [9, 1]])
s.plot_trajectory("self.trajectory_2_parameters(H=10, v0x=20)")
s.plot_pendulum("self.pendulum_disk_rotating_axis_at_its_edge(10, 10, 10)")