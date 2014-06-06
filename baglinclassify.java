import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;

/**
 * Bootstrap aggregating linear classifier.
 * @author AbstractOwl <>
 */
public class baglinclassify {
	private static final DecimalFormat FMT = new DecimalFormat("0.0");
	
	/**
	 * Linear classifier.
	 * @author AbstractOwl <>
	 */
	static class linclassify {
		private int m_p; // number of positive samples
		private int m_n; // number of negative samples
		private int n;   // len of vectors
		
		private double t; // threshold
		
		private double[] centroid_p; // centroid of positive samples
		private double[] centroid_n; // centroid of negative samples
		private double[] w;          // discriminant
		
		private double[][] samples_p;
		private double[][] samples_n;
		
		public linclassify(int m_p, int m_n, int n) {
			this.m_p = m_p;
			this.m_n = m_n;
			this.n   = n;

			centroid_p = null;
			centroid_n = null;
			
			samples_p = null;
			samples_n = null;
			
			w = new double[n];
		}
		
		/**
		 * Computes the centroid of a given set of vectors.
		 * 
		 * @param vectors Set of vectors to compute on
		 * @param m       Number of samples to expect 
		 * @param n       Number of features to expect
		 * 
		 * @return Centroid vector of size n
		 */
		private double[] calculateCentroid(double[][] vectors, int m, int n) {
			if (vectors.length != m) {
				throw new IllegalArgumentException(
					"Training set dimensions don't match (m, n)"
				);
			}
			
			// Short circuit if no samples
			if (m == 0) {
				return new double[n];
			}
			
			if (vectors[0].length != n) {
				throw new IllegalArgumentException(
					"Training set dimensions don't match (m, n)"
				);
			}
			double[] centroid = new double[n];
			
			for (int i = 0; i < m; ++i) { 
				for (int j = 0; j < n; ++j) {
					centroid[j] += vectors[i][j];
				}
			}
			
			for (int i = 0; i < n; ++i) {
				centroid[i] /= m;
			}
			
			return centroid;
		}
		/**
		 * Prints the sample set.
		 */
		public void output() {
			StringBuilder sb = new StringBuilder();
			
			for (int i = 0; i < samples_p.length; ++i) {
				for (int j = 0; j < samples_p[0].length; ++j) {
					sb.append(FMT.format(samples_p[i][j])).append(' ');
				}
				sb.append("- True\n");
			}
			for (int i = 0; i < samples_n.length; ++i) {
				for (int j = 0; j < samples_n[0].length; ++j) {
					sb.append(FMT.format(samples_n[i][j])).append(' ');
				}
				sb.append("- False\n");
			}
			System.out.print(sb.toString());
		}
		/**
		 * Trains the linear classifier with the given dataset.
		 * 
		 * @param vectors_p Vectors in positive dataset
		 * @param vectors_n Vectors in negative dataset
		 */
		public void train(double[][] vectors_p, double[][] vectors_n) {
			samples_p = vectors_p;
			samples_n = vectors_n;
			
			m_p = vectors_p.length;
			m_n = vectors_n.length;
			
			centroid_p = calculateCentroid(vectors_p, m_p, n);
			centroid_n = calculateCentroid(vectors_n, m_n, n);
			
			// Calculate discriminant
			for (int i = 0; i < n; ++i) {
				w[i] = centroid_p[i] - centroid_n[i];
			}
			
			// Calculate t
			for (int i = 0; i < n; ++i) {
				t += w[i] * (centroid_p[i] + centroid_n[i]);
			}
			t /= 2;
		}
		/**
		 * Classifies a vector based on learning from training.
		 * @param   vector Vector to test
		 * @return  true or false depending on positive or negative
		 *          classification
		 */
		public boolean test(double[] vector) {
			if (centroid_p == null) {
				throw new IllegalStateException("Please run train first");
			}
			if (vector.length != n) {
				throw new IllegalArgumentException(
						"Testing vector does not match expected len"
						);
			}
			
			if (samples_p.length == 0) {
				return false;
			} else if (samples_n.length == 0) {
				return true;
			}
			
			double product = 0.0;
			for (int i = 0; i < n; ++i) {
				product += w[i] * vector[i];
			}
			return product >= t;
		}
		
		/*
		private static void test_linclassify() {
			linclassify l = new linclassify(3, 3, 2);
			l.train(new double[][] {{5, 5}, {6, 6}, {7, 7}}, new double[][] {{0, 0}, {1, 1}, {2, 2}});
			if (!	(l.test(new double[] {0, 0}) == false)
				&&	(l.test(new double[] {6, 6}) == true)
				&&	(l.test(new double[] {3, 3}) == false)
				) {
					throw new RuntimeException("Test failed.");
			}
		}
		*/
	}
	public static void main(String args[]) throws IOException {
		boolean verbose = (args.length == 5 && args[0].equals("-v"));
		int offset = verbose ? 1 : 0;
		
		Random g = new Random();
		
		if (args.length - offset != 4) {
			throw new IllegalArgumentException(
				"usage: baglinclassify [-v] T size <train> <test>"
			);
		}
		
		// T:    Number of classifiers
		// size: Number of samples 
		int T    = Integer.parseInt(args[0 + offset], 10);
		int size = Integer.parseInt(args[1 + offset], 10);
		
		// train: training data set
		// test:  testing data set
		String train = args[2 + offset];
		String test  = args[3 + offset];
		
		BufferedReader reader = new BufferedReader(new FileReader(train));
		
		String[] header = reader.readLine().split("\\s+");
		
		int m   = Integer.parseInt(header[0], 10);
		int n_p = Integer.parseInt(header[1], 10);
		int n_n = Integer.parseInt(header[2], 10);
		
		double[][] vectors_p = new double[n_p][m];
		double[][] vectors_n = new double[n_n][m];
		
		// Parse positive vectors
		for (int i = 0; i < n_p; ++i) {
			String[] features = reader.readLine().trim().split("\\s+");
			if (features.length != m) {
				reader.close();
				throw new IllegalArgumentException("Feature length mismatch");
			}
			for (int j = 0; j < m; ++j) {
				vectors_p[i][j] = Double.parseDouble(features[j]);
			}
		}
		
		// Parse negative vectors
		for (int i = 0; i < n_n; ++i) {
			String[] features = reader.readLine().trim().split("\\s+");
			if (features.length != m) {
				reader.close();
				throw new IllegalArgumentException("Feature length mismatch");
			}
			for (int j = 0; j < m; ++j) {
				vectors_n[i][j] = Double.parseDouble(features[j]);
			}
		}
		
		// Construct classifiers
		linclassify[] classifiers = new linclassify[T];
		for (int i = 0; i < T; ++i) {
			int threshold = g.nextInt(size + 1);
			double[][] sample_p = new double[threshold][m];
			for (int j = 0; j < threshold; ++j) {
				System.arraycopy(
					vectors_p[g.nextInt(vectors_p.length)],
					0,
					sample_p[j],
					0,
					m
				);
			}
			double[][] sample_n = new double[size - threshold][m];
			for (int j = 0; j < size - threshold; ++j) {
				System.arraycopy(
					vectors_n[g.nextInt(vectors_n.length)],
					0,
					sample_n[j],
					0,
					m
				);
			}
			
			classifiers[i] = new linclassify(threshold, size - threshold, m);
			classifiers[i].train(sample_p, sample_n);
		}
		reader.close();
		
		// Parse testing data
		reader = new BufferedReader(new FileReader(test));
		header = reader.readLine().split("\\s+");
		
		if (m != Integer.parseInt(header[0])) {
			throw new IllegalArgumentException("Feature count mismatch");
		}
		
		n_p = Integer.parseInt(header[1], 10);
		n_n = Integer.parseInt(header[2], 10);

		int fp, fn;
		fp = fn = 0;
		StringBuilder sb = new StringBuilder();
		sb.append("Classification:\n");
		for (int i = 0, total = n_p + n_n; i < total; ++i) {
			double[] vector = new double[m];
			String[] features = reader.readLine().trim().split("\\s+");
			
			for (int j = 0; j < m; ++j) {
				vector[j] = Double.parseDouble(features[j]);
			}
			
			int[] results = new int[T];
			for (int j = 0; j < T; ++j) {
				results[classifiers[j].test(vector) ? 1 : 0]++;
			}
			
			boolean outcome = results[1] >= results[0];
			
			// Print
			for (int j = 0; j < vector.length; ++j) {
				sb.append(FMT.format(vector[j])).append(' ');
			}
			sb.append("- ").append(outcome ? "True" : "False");
			if (outcome == i < n_p) {
				sb.append(" (correct)\n");
			} else {
				if (outcome) {
					sb.append(" (false positive)\n");
					fp++;
				} else {
					sb.append(" (false negative)\n");
					fn++;
				}
			}
		}
		System.out.format("Positive examples: %d\n", n_p);
		System.out.format("Negative examples: %d\n", n_n);
		System.out.format("False positives: %d\n", fp);
		System.out.format("False negatives: %d\n", fn);
		
		System.out.println();
		
		if (verbose) {
			for (int i = 0; i < T; ++i) {
				System.out.format("Bootstrap sample set %d:\n", i + 1);
				classifiers[i].output();
				System.out.println();
			}
			
			System.out.print(sb.toString());
		}
	}
}
