import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class JDBCUtil {
    private static final String DRIVER = "org.postgresql.Driver";

    // DB connection is read from the standard libpq env vars so no local
    // account is hardcoded; override via PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD.
    private static String env(String key, String def) {
        String v = System.getenv(key);
        return (v == null || v.isEmpty()) ? def : v;
    }

    private static final String URL = "jdbc:postgresql://" + env("PGHOST", "localhost")
        + ":" + env("PGPORT", "5432") + "/" + env("PGDATABASE", "postgres");
    private static final String NAME = env("PGUSER", "postgres");
    private static final String PASSWORD = env("PGPASSWORD", "");
    private static Connection conn = null;


    static{
        try {
            Class.forName(DRIVER);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }


    public static Connection getConnection() {
        try {
            return DriverManager.getConnection(URL, NAME, PASSWORD);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

}
