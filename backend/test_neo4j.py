from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import time

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "testpassword"
    
    print(f"Attempting to connect to Neo4j at {uri}")
    
    # Try multiple times with delay
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1} of {max_attempts}")
            
            # Create driver instance
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Verify connection
            with driver.session() as session:
                result = session.run("RETURN 1 as n")
                record = result.single()
                if record and record["n"] == 1:
                    print("✅ Successfully connected to Neo4j!")
                    
                    # Test creating a simple node
                    session.run("CREATE (n:Test {name: 'test'})")
                    print("✅ Successfully created a test node")
                    
                    # Clean up test node
                    session.run("MATCH (n:Test) DELETE n")
                    print("✅ Successfully cleaned up test node")
                    
                    driver.close()
                    return True
                else:
                    print("❌ Connection test failed")
            
            driver.close()
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                print("Waiting 5 seconds before next attempt...")
                time.sleep(5)
            else:
                print("All connection attempts failed")
                return False
    
    return False

if __name__ == "__main__":
    test_connection() 