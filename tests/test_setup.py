"""Fase 0: el esqueleto importa completo."""


def test_importa_paquetes():
    import data.cache
    import data.fred
    import data.panel
    import data.prices
    import data.sec
    import data.universe
    import model.black_scholes
    import model.merton_iterativo
    import model.naive  # noqa: F401
