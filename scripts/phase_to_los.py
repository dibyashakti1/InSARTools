from insartools.phase import PhaseConverter

def main():

    converter = PhaseConverter(
        input_file=args.input,
        output_file=args.output,
        sensor=args.sensor,
        wavelength=args.wavelength,
        logfile=args.log,
    )

    converter.run()


if __name__ == "__main__":
    main()
