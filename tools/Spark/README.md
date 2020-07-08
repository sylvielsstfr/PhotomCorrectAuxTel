# README.md on lauching PySpark with spark-fit


## Installation of Spark

    in /Users/dagoret/MacOSX/External/Spark/spark-2.4.6-bin-hadoop2.7

- put bin path in PATH   

## Configureation
    export PYSPARK_PYTHON=python
    export PYSPARK_DRIVER_PYTHON=jupyter
    export PYSPARK_DRIVER_PYTHON_OPTS='notebook' 		



## Launch pyspack with jupyter notebook


    (base) /Users/dagoret/MacOSX/GitHub/LSST/PhotomCorrectAuxTel/tools/Spark>

    pyspark --packages com.github.astrolabsoftware:spark-fits_2.11:0.6.0 


## in notebook

         df = spark.read\
         .format("fits")\
         .option("hdu",1)\
         .load("path_to_fits_filename")

         df.show()

         df.printSchema() 	
