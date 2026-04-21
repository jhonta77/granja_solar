"""Lanzador del EDA academico para el proyecto de granja solar."""


def run() -> int:
    """Carga el script principal y muestra errores de dependencias de forma legible."""

    try:
        from notebooks.eda_fuente_xm import main
    except ModuleNotFoundError as error:
        missing_name = getattr(error, "name", "desconocido")
        print(
            "Falta una dependencia de Python para ejecutar el EDA. "
            f"Modulo faltante: {missing_name}. "
            "Instala o activa un entorno con pandas, matplotlib y openpyxl."
        )
        return 1

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
