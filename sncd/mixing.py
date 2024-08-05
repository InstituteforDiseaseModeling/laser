import numpy as np


def pairwise_haversine(df):
    """pairwise distances for all (lon, lat) points"""

    earth_radius_km = 6367

    data = df[["Long", "Lat"]].map(np.radians).to_numpy()  # N.B. conversion to radians
    lon = data[:, 0]
    lat = data[:, 1]

    # matrices of pairwise differences for latitudes & longitudes
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon

    # vectorized haversine distance calculation
    d = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


def init_gravity_diffusion(df, scale, dist_exp):
    if len(df) == 1:
        return np.ones((1, 1))

    distances = pairwise_haversine(df)

    pops = df.population.values
    pops = pops[:, np.newaxis].T
    pops = np.repeat(pops, pops.size, axis=0).astype(np.float64)

    np.fill_diagonal(distances, 100000000)  # Prevent divide by zero errors and self migration
    diffusion_matrix = pops / (distances + 10) ** dist_exp  # minimum distance prevents excessive neighbor migration
    np.fill_diagonal(diffusion_matrix, 0)

    # normalize average total outbound migration to 1
    diffusion_matrix = diffusion_matrix / np.mean(np.sum(diffusion_matrix, axis=1))

    diffusion_matrix *= scale
    diagonal = 1 - np.sum(diffusion_matrix, axis=1)  # normalized outbound migration by source
    np.fill_diagonal(diffusion_matrix, diagonal)

    return diffusion_matrix


if __name__ == "__main__":
    from settlements import parse_settlements

    settlements_df = parse_settlements()

    distances_km = pairwise_haversine(settlements_df)
    # print(distances_km)

    M = init_gravity_diffusion(settlements_df, 0.01, 1.5)
    # print(M)

    # --------

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    def plot_flux(df, M, source, ax):
        flux = M[df.index.get_loc(source), :]
        df["flux"] = flux
        df.loc[source, "flux"] = np.nan
        df.plot(kind="scatter", x="Long", y="Lat", s=0.1 * np.sqrt(df.population), alpha=0.5, c="flux", norm=LogNorm(), ax=ax)
        ax.set_title(f"{source} flux")

    plot_flux(settlements_df, M, "London", axs[0])
    plot_flux(settlements_df, M, "Newcastle", axs[1])
    fig.set_tight_layout(True)

    plt.show()
