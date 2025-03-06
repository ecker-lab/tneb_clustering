import numpy as np
import itertools

COLOR_DICT = {
    "blue": "#00549F",  # RWTH-Blau, die Hausfarbe
    "blue_75": "#407FB7",
    "blue_50": "#8EBAE5",
    "blue_25": "#C7DDF2",
    "blue_10": "#E8F1FA",
    "black": "#000000",  # sekundärfarbe
    "black_75": "#646567",
    "black_50": "#9C9E9F",
    "black_25": "#CFD1D2",
    "black_10": "#ECEDED",
    "magenta": "#E30066",  # sekundärfarbe
    "magenta_75": "#E96088",
    "magenta_50": "#F19EB1",
    "magenta_25": "#F9D2DA",
    "magenta_10": "#FDEEF0",
    "yellow": "#FFED00",  # sekundärfarbe
    "yellow_75": "#FFF055",
    "yellow_50": "#FFF59B",
    "yellow_25": "#FFFAD1",
    "yellow_10": "#FFFDEE",
    # ergänzende Farben
    "petrol": "#006165",
    "petrol_75": "#2D7F83",
    "petrol_50": "#7DA4A7",
    "petrol_25": "#BFD0D1",
    "petrol_10": "#E6ECEC",
    "turquoise": "#0098A1",
    "turquoise_75": "#00B1B7",
    "turquoise_50": "#89CCCF",
    "turquoise_25": "#CAE7E7",
    "turquoise_10": "#EBF6F6",
    "green": "#57AB27",  # grün wie in der Uniklinik
    "green_75": "#8DC060",
    "green_50": "#B8D698",
    "green_25": "#DDEBCE",
    "green_10": "#F2F7EC",
    "lime": "#BDCD00",
    "lime_75": "#D0D95C",
    "lime_50": "#E0E69A",
    "lime_25": "#F0F3D0",
    "lime_10": "#F9FAED",
    "orange": "#F6A800",
    "orange_75": "#FABE50",
    "orange_50": "#FDD48F",
    "orange_25": "#FEEAC9",
    "orange_10": "#FFF7EA",
    "red": "#CC071E",
    "red_75": "#D85C41",
    "red_50": "#E69679",
    "red_25": "#F3CDBB",
    "red_10": "#FAEBE3",
    "bordeaux": "#A11035",
    "bordeaux_75": "#B65256",
    "bordeaux_50": "#CD8B87",
    "bordeaux_25": "#E5C5C0",
    "bordeaux_10": "#F5E8E5",
    "purple": "#612158",
    "purple_75": "#834E75",
    "purple_50": "#A8859E",
    "purple_25": "#D2C0CD",
    "purple_10": "#EDE5EA",
    "lila": "#7A6FAC",  # die englische Variante ist ebenfalls purple
    "lila_75": "#9B91C1",
    "lila_50": "#BCB5D7",
    "lila_25": "#DEDAEB",
    "lila_10": "#F2F0F7",
}

def get_color_scheme(n_colors):
    colors = np.array(
        list(
            itertools.islice(
                itertools.cycle(
                    [
                        "#377eb8",  # blue
                        "#ff7f00",  # orange
                        "#4daf4a",  # green
                        "#f781bf",  # pink
                        "#a65628",  # brown
                        "#984ea3",  # purple
                        "#999999",  # grey
                        "#e41a1c",  # red
                        "#dede00",  # yellow
                        "#ADD8E6",  # light blue,added by Martin (or 87CEEB)
                        # "#00BFFF",  # was in `get_color_scheme_from_preds` instead of the line before
                    ]
                ),
                int(n_colors),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    return colors