import pandas as pd
from scipy.stats import genpareto
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
mu_0 = 1.25663706212e-6  # Permeability of Free Space (H/m)

class pyExtremeHelper:
    def ler_arquivo_dat(nome_arquivo):
        with open(nome_arquivo, "r") as arquivo:
            conteudo = arquivo.readlines()  # Lê todas as linhas do arquivo
            
            # Encontra a primeira linha contendo START_VARIABLE = nome_variavel
            var = ""
            for linha in conteudo:
                if "START_VARIABLE" in linha:
                    var = linha.split("=")[1].strip()
                    break  # Sai do loop quando encontrar a linha desejada
                
            # Procura por linhas começando com SIZES = e adiciona as variáveis ao array
            variaveis = []
            for linha in conteudo:
                if "SIZES" in linha:
                    quantidade = int(linha.split("=")[1])
                    for i in range(1, quantidade + 1):
                        variaveis.append(f"{var}{i}")
                elif "START_VARIABLE" in linha:
                    var = linha.split("=")[1].strip()
                if "EOF" in linha:
                    break  # Sai do loop quando encontrar EOF
                
            # Cria um dataframe pandas a partir das linhas restantes do arquivo
            dados = [linha.strip().split(",") for linha in conteudo if len(linha.strip().split(",")) == 11]
            df = pd.DataFrame(dados, columns=variaveis)
            return df
    
    def calculate_B_diff(df1, df2):
        '''
        Calculate the magnetic field difference between two satellites.
        '''
        B_diff = df1[['B_vec_xyz_gse__C1_CP_FGM_FULL1','B_vec_xyz_gse__C1_CP_FGM_FULL2', 'B_vec_xyz_gse__C1_CP_FGM_FULL3']].values - df2[['B_vec_xyz_gse__C1_CP_FGM_FULL1','B_vec_xyz_gse__C1_CP_FGM_FULL2', 'B_vec_xyz_gse__C1_CP_FGM_FULL3']].values
        return B_diff
    
    def calculate_r_diff(df1, df2):
        '''
        Calculate the distance between two satellites.
        '''
        r_diff = df1[['sc_pos_xyz_gse__C1_CP_FGM_FULL1', 'sc_pos_xyz_gse__C1_CP_FGM_FULL2', 'sc_pos_xyz_gse__C1_CP_FGM_FULL3']].values - df2[['sc_pos_xyz_gse__C1_CP_FGM_FULL1', 'sc_pos_xyz_gse__C1_CP_FGM_FULL2', 'sc_pos_xyz_gse__C1_CP_FGM_FULL3']].values
        return r_diff
    
    def calculate_current_density(df1, df2, df3):
        '''
        Calculate the total current density using equation (2)
        '''
        r12 = calculate_r_diff(df1, df2)
        r13 = calculate_r_diff(df1, df3)
        r23 = calculate_r_diff(df2, df3)
    
        B12 = calculate_B_diff(df1, df2)
        B13 = calculate_B_diff(df1, df3)
        B23 = calculate_B_diff(df2, df3)
        
        temp = pd.DataFrame()
        temp['r13'] = r13.tolist()
        temp['r23'] = r23.tolist()
        temp['B13'] = B13.tolist()
        temp['B23'] = B23.tolist()
    
        Jijk = temp.apply(lambda row: (1/mu_0) * ((np.dot(row['B13'], row['r23']) - np.dot(row['B23'], row['r13']))/ np.dot(np.cross(row['r13'], row['r23']), np.cross(row['r13'], row['r23']))), axis=1)
    
        return Jijk
    def curlometer(spacecraft1, spacecraft2, spacecraft3, spacecraft4):
        '''
        Calculate the curlometer current density using equation (2)
        '''
        J123 = calculate_current_density(spacecraft1, spacecraft2, spacecraft3)
        J124 = calculate_current_density(spacecraft1, spacecraft2, spacecraft4)
        J134 = calculate_current_density(spacecraft1, spacecraft3, spacecraft4)
        J234 = calculate_current_density(spacecraft2, spacecraft3, spacecraft4)
        return (((J123 + J124 + J134 + J234)/4)**2)**0.5
    
    def calculate_mod_B(df, Bx_column, By_column, Bz_column):
        """
        Function to calculate mod_B from the given pandas dataframe and column names.
        Args:
        df : pandas.DataFrame
        Bx_column, By_column, Bz_column : str
    
        Returns:
        pandas.Series
        """
        # Get the columns
        Bx = df[Bx_column]
        By = df[By_column]
        Bz = df[Bz_column]
    
        # Calculate the magnitude of B (mod_B)
        yy = np.sqrt(Bx**2 + By**2 + Bz**2)
    
        return yy
    
    # Util function to plot the mod_B
    def plot_mod_B(yy):
        plt.figure()
        plt.plot(yy)
        plt.show()
    
    
    def calculate_PVI(x, tau=66):
        """
        Function to calculate PVI from the given pandas series
        Args:
        x : pandas.Series
        tau : int
    
        Returns:
        pandas.Series
        """
        
        # Ensure x is a pandas series
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
    
        n = len(x)
        delta = []
    
        # Loop through to compute delta
        for i in range(0, n-tau):
            delta.append(x[i+tau] - x[i])
    
        delta = pd.Series(delta)
    
        # Compute absolute, square, mean, and root
        abs_delta = delta.abs()
        square_delta = abs_delta.pow(2)
        mean_delta = square_delta.mean()
        sqrt_delta = np.sqrt(mean_delta)
    
        # Compute PVI
        PVI = abs_delta.divide(sqrt_delta)
    
        # Transpose
        PVI = PVI.T
    
        return PVI
    
    # Using matplotlib for plotting
    def plot_data(PVI):
        plt.figure()
        plt.plot(PVI)
        plt.show()
    
    
    def norm(vector):
        return np.sqrt(np.sum(np.square(vector)))
    
    def dot(v1, v2):
        return np.dot(v1, v2)
        
    def angle(v1, v2):
        v1_norm = norm(v1)
        v2_norm = norm(v2)
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot_product = dot(v1, v2)
        
        return 180.0 * np.arccos(dot_product) / np.pi
    
    def cs_detection(df, tau, theta_c):
        b1 = df.iloc[0:tau].values
        b2 = df.iloc[tau:2*tau].values
        
        ff_c = 0.15
        cont = 0
        for i in range(tau):
            theta = angle(b1[i], b2[i])
            if theta >= theta_c:
                cont +=1
        
        ff = float(cont)/float(tau)
        
        if ff >= ff_c:
            out = 1
        else:
            out = 0
        
        return out
    
    def limethod(df, theta_c = 35.0, tau_sec = 10):
        
        # Calculate timesteps
        dt = 1.0 / 22.0
        tau = int(22 * tau_sec)
        
        outputs = []
        data_points = df.values
        for i in range(tau, len(data_points) - tau):
            window = df.iloc[i-tau+1 : i+tau+1]
            cs_out = cs_detection(window, tau, theta_c)
            outputs.append([df.index[i], cs_out])
            
        detected_df = pd.DataFrame(outputs, columns=['Time', 'cs_out'])
        
        return detected_df
    
    def convert_to_float(s):
        try:
            return float(s)
        except ValueError:
            # Handle specific formatting issues if necessary
            return float(s.replace('D', 'E'))
    
    def calculate_magnetic_volatility(df, B, tau=50, w=50):
        """
        Calcular a volatilidade magnética.
    
        Parâmetros:
        df (pandas.DataFrame): DataFrame contendo os dados.
        B (str): Nome da coluna contendo o campo magnético.
        tau (int, opcional): Valor a ser usado para τ.
        w (int, opcional): Tamanho da janela para calcular a volatilidade magnética.
    
        Retorna:
        vol_mag (pandas.Series): Volatilidade magnética calculada.
        """
        
        # Calcular r_mag(t)
        #df['r_mag'] = np.log(df[B].abs())
    
        # Calcular Delta_r_mag(t)
        #df['Delta_r_mag'] = df['r_mag'].shift(periods=tau) - df['r_mag']
    
        df['Delta_r_mag'] = np.log(df[B].shift(periods=tau) / df[B])
        # Dropout do NaN produzidos pelo shiftment
        df = df.dropna()
    
        # Calcular vol_mag(t)
        df['vol_mag'] = df['Delta_r_mag'].rolling(window=w).std()
        
        return df['vol_mag']
    
    def apply_gaussian_kernel(x_coords, sigma):
        smoothed_x = gaussian_filter1d(x_coords, sigma)
        return smoothed_x


    def declustering_function(data, u=30000, run=10):
        # Threshold and run are parameters for the function with default values
        # Compute the POT (Peaks Over Threshold)
        pot_df = data[data['value'] > u].copy()
        pot_df['index'] = pot_df.index

        # Compute the VBT (Values Below Threshold) - for completeness
        vbt_df = data[data['value'] < u].copy()

        # Create a new column 'cluster' in the POT data frame
        # A new cluster starts if the gap between the positions is larger than 'run'
        pot_df['cluster'] = (pot_df['index'] - pot_df['index'].shift() > run).cumsum()

        # Compute the declustered POT (maximum value in each cluster)
        decluster_df = pot_df.groupby('cluster')['value'].idxmax()
        declustered_pot = data.loc[decluster_df]

        # Plotting
        # separate the points into two groups using the threshold
        below_threshold = data[data['value'] < u]
        above_threshold = data[data['value'] >= u]

        # plot points below threshold in black
        plt.scatter(below_threshold.index, below_threshold['value'], color='black')

        # plot points above threshold in transparent gray
        plt.scatter(above_threshold.index, above_threshold['value'], color='gray', alpha=0.5)

        # plot declustered points in red
        plt.scatter(declustered_pot.index, declustered_pot['value'], color='red')

        # plot the threshold line
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = u * np.ones_like(x_vals)
        plt.plot(x_vals, y_vals, color='red', ls='--')
        plt.show()

        # Output
        print('Number of Clusters =:', declustered_pot.shape[0])
        return declustered_pot

        # A new cluster starts if the gap between the positions is larger than 'run'
        pot_df['cluster'] = (pot_df['index'] - pot_df['index'].shift() > run).cumsum()

        # Compute the declustered POT (maximum value in each cluster)
        decluster_df = pot_df.groupby('cluster')['value'].idxmax()
        declustered_pot = data.loc[decluster_df]

        # Plotting
        # separate the points into two groups using the threshold
        below_threshold = data[data['value'] < u]
        above_threshold = data[data['value'] >= u]

        # plot points below threshold in black
        plt.scatter(below_threshold.index, below_threshold['value'], color='black')

        # plot points above threshold in transparent gray
        plt.scatter(above_threshold.index, above_threshold['value'], color='gray', alpha=0.5)

        # plot declustered points in red
        plt.scatter(declustered_pot.index, declustered_pot['value'], color='red')

        # plot the threshold line
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = u * np.ones_like(x_vals)
        plt.plot(x_vals, y_vals, color='red', ls='--')
        plt.show()

        # Output
        print('Number of Clusters =:', declustered_pot.shape[0])
        return declustered_pot


    # %%
    # Define the mean and standard deviation of excess
    def stats_excess(data, threshold):
        # Extract values above the threshold
        excess = data[data > threshold]

        # Calculate the mean excess
        mean_excess = np.mean(excess - threshold)

        # Calculate the standard deviation of excess
        std_excess = np.std(excess - threshold)

        return mean_excess, std_excess

    # Function to plot mean excess for multiple thresholds
    def plot_mean_excess(data, min_thresh, max_thresh, num_threshs=100):
        # Create a numpy array of thresholds from min_thresh to max_thresh
        thresholds = np.linspace(min_thresh, max_thresh, num_threshs)

        # Calculate mean excess and its standard deviation for each threshold
        mean_excesses, std_excesses = zip(*[stats_excess(data, thresh) for thresh in thresholds])

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Draw the plot along with error bars
        ax.errorbar(thresholds, mean_excesses, yerr= std_excesses, fmt='o', label='Mean Excess')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        ax.legend(loc='best')

        # Show the plot
        plt.show()

    def fit_pot_model(data, min_threshold, max_threshold, num_thresholds):
        """
        This function fits a Peaks Over Threshold (POT) model to a range of thresholds.
        :param data: A pandas Series or DataFrame.
        :param min_threshold: The minimum threshold value.
        :param max_threshold: The maximum threshold value.
        :param num_thresholds: The number of thresholds.
        :return: A DataFrame with the fitted parameters of the POT model for each threshold.
        """
        results = []
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

        for threshold in thresholds:
            # Select data over the threshold
            exceedances = data[data > threshold] - threshold

            # Fit the Generalized Pareto Distribution (GPD) to the exceedances
            c, loc, scale = genpareto.fit(exceedances)

            results.append({
                'Threshold': threshold,
                'Shape': c,
                'Location': loc,
                'Scale': scale
            })

        return pd.DataFrame(results)

    def plot_shape_parameter(results):
        """
        This function plots the 'Shape' parameter of the Peaks Over Threshold (POT) model.
        :param results: A DataFrame with the fitted parameters of the POT model for each threshold.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(results['Threshold'], results['Shape'], marker='o')
        plt.title('Evolution of Shape Parameter over Thresholds')
        plt.xlabel('Threshold')
        plt.ylabel('Shape Parameter')
        plt.grid(True)
        plt.show()

    # Assume 'results' is the DataFrame returned by 'fit_pot_model'
    # You may need to run `fit_pot_model` function prior to plot

    def plot_mean_residual_life(data, thresholds):
        plt.figure(figsize = (10, 5))
        means = []

        for threshold in thresholds:
            means.append(np.mean(data[data > threshold] - threshold))

        derivatives = np.gradient(means)
        derivatives_change_rate = np.abs(np.gradient(derivatives))

        approx_const_start_index = np.where(derivatives_change_rate < 0.01)[0][0]
        approx_const_start_threshold = thresholds[approx_const_start_index]

        plt.plot(thresholds, means, marker='o')
        plt.axvline(x=approx_const_start_threshold, color='r', linestyle='--')
        plt.text(approx_const_start_threshold, max(means)/2, f'Linear begin ~{approx_const_start_threshold}', color='r')

        plt.grid(True)
        plt.xlabel('Thresholds')
        plt.ylabel('Mean Excess')
        plt.title('Mean residual life plot')
        plt.show()
