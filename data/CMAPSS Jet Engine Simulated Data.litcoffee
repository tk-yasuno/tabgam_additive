CMAPSS Jet Engine Simulated Data

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to: 1) unit number 2) time, in cycles 3) operational setting 1 4) operational setting 2 5) operational setting 3 6) sensor measurement 1 7) sensor measurement 2 ... 26) sensor measurement 26

Data Set: FD001 Train trjectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: ONE (HPC Degradation)

Data Set: FD002 Train trjectories: 260 Test trajectories: 259 Conditions: SIX Fault Modes: ONE (HPC Degradation)

Data Set: FD003 Train trjectories: 100 Test trajectories: 100 Conditions: ONE (Sea Level) Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004 Train trjectories: 248 Test trajectories: 249 Conditions: SIX Fault Modes: TWO (HPC Degradation, Fan Degradation)

Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, ‘Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation’, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

Additional Info
Field	Value
Maintainer	Chris Teubert
Last Updated	May 30, 2025, 5:34 AM (UTC+09:00)
Created	April 1, 2025, 3:46 AM (UTC+09:00)
accessLevel	public
bureauCode	026:00
catalog_@context	https://project-open-data.cio.gov/v1.1/schema/catalog.jsonld
catalog_@id	https://data.nasa.gov/data.json
catalog_conformsTo	https://project-open-data.cio.gov/v1.1/schema
catalog_describedBy	https://project-open-data.cio.gov/v1.1/schema/catalog.json
harvest_object_id	b80ff4d6-685e-4377-bc5e-4dac60c71d0b
harvest_source_id	61638e72-b36c-4866-9d28-551a3062f158
harvest_source_title	DNG Legacy Data
identifier	https://data.nasa.gov/api/views/ff5v-kuh6
issued	2022-07-14
landingPage	https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
modified	2024-05-15
programCode	026:021
publisher	PCoE
resource-type	Dataset
source_datajson_identifier	true
source_hash	eccb6957e4312dd2bbde3a2d3579a0d479b5893f9338e898d9cae1b774eb5c1b
source_schema_version	1.1
theme	Aerospace


