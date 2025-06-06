# This module provides a service for interacting with a Neo4j graph database.
# It includes methods for connecting to the database, creating constraints, and managing student and class nodes and relationships.

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Relationship type constants
RELATIONSHIP_TYPES = {
    "FRIEND": "friend",
    "NEUTRAL": "neutral",
    "CONFLICT": "conflict"
}

class Neo4jService:
    def __init__(self):
        # Initialize connection parameters for the Neo4j database.
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "testpassword"
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        """Create necessary constraints in the database to ensure data integrity."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Student) REQUIRE s.student_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.class_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            self.run_query(constraint)

    def run_query(self, query, parameters=None):
        """Execute a Cypher query against the Neo4j database."""
        if not self.driver:
            self.connect()
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record for record in result]
        except Exception as e:
            print(f"Query failed: {str(e)}")
            return None

    def create_student(self, student_id, name, preferences=None):
        """Create a student node with optional preferences."""
        query = """
        MERGE (s:Student {student_id: $student_id})
        SET s.name = $name
        """
        if preferences:
            query += ", s.preferences = $preferences"
        
        return self.run_query(query, {
            "student_id": student_id,
            "name": name,
            "preferences": preferences
        })

    def create_class(self, class_id, name, capacity):
        """Create a class node with specified attributes."""
        query = """
        MERGE (c:Class {class_id: $class_id})
        SET c.name = $name,
            c.capacity = $capacity
        """
        return self.run_query(query, {
            "class_id": class_id,
            "name": name,
            "capacity": capacity
        })

    def allocate_student_to_class(self, student_id, class_id):
        """Allocate a student to a class by creating an ENROLLED_IN relationship."""
        query = """
        MATCH (s:Student {student_id: $student_id})
        MATCH (c:Class {class_id: $class_id})
        MERGE (s)-[:ENROLLED_IN]->(c)
        """
        return self.run_query(query, {
            "student_id": student_id,
            "class_id": class_id
        })

    def get_students_in_class(self, class_id):
        """Get all students allocated to a specific class."""
        query = """
        MATCH (s:Student)-[:ENROLLED_IN]->(c:Class {class_id: $class_id})
        RETURN s.student_id as student_id, s.name as name
        """
        return self.run_query(query, {"class_id": class_id})

    def get_student_classes(self, student_id):
        """Get all classes a student is enrolled in."""
        query = """
        MATCH (s:Student {student_id: $student_id})-[:ENROLLED_IN]->(c:Class)
        RETURN c.class_id as class_id, c.name as name, c.capacity as capacity
        """
        return self.run_query(query, {"student_id": student_id})

    def get_class_capacity(self, class_id):
        """Get the current enrollment and capacity of a class."""
        query = """
        MATCH (c:Class {class_id: $class_id})
        OPTIONAL MATCH (s:Student)-[:ENROLLED_IN]->(c)
        RETURN c.capacity as capacity, count(s) as current_enrollment
        """
        return self.run_query(query, {"class_id": class_id})

    def remove_student_from_class(self, student_id, class_id):
        """Remove a student's allocation from a class by deleting the ENROLLED_IN relationship."""
        query = """
        MATCH (s:Student {student_id: $student_id})-[r:ENROLLED_IN]->(c:Class {class_id: $class_id})
        DELETE r
        """
        return self.run_query(query, {
            "student_id": student_id,
            "class_id": class_id
        })

    def get_all_allocations(self):
        """Get all student-class allocations."""
        query = """
        MATCH (s:Student)-[:ENROLLED_IN]->(c:Class)
        RETURN s.student_id as student_id, s.name as student_name,
               c.class_id as class_id, c.name as class_name
        """
        return self.run_query(query)

    def create_relationship(self, student1_id, student2_id, relationship_type, attention=0.5):
        """Create a relationship between two students with specified type and attention value."""
        if relationship_type not in RELATIONSHIP_TYPES.values():
            raise ValueError(f"Invalid relationship type: {relationship_type}. Must be one of: {list(RELATIONSHIP_TYPES.values())}")
            
        query = """
        MATCH (s1:Student {student_id: $student1_id})
        MATCH (s2:Student {student_id: $student2_id})
        MERGE (s1)-[r:RELATES_TO {
            type: $relationship_type,
            attention: $attention
        }]->(s2)
        """
        return self.run_query(query, {
            "student1_id": student1_id,
            "student2_id": student2_id,
            "relationship_type": relationship_type,
            "attention": attention
        })

    def export_d3_graph(self):
        """Export graph data in D3.js format for visualization."""
        query = """
        MATCH (s:Student)-[r:RELATES_TO]->(t:Student)
        RETURN s.student_id AS source, t.student_id AS target, r.type AS type, r.attention AS attention
        """
        result = self.run_query(query)
        if result and len(result) > 0:
            links = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "type": record.get("type", ""),
                    "attention": record.get("attention", 0)
                }
                for record in result
            ]

            # Query for all students (nodes)
            node_query = """
            MATCH (s:Student)
            RETURN s.student_id AS id,
                   s.class_id AS group,
                   s.achievement AS achievement,
                   s.wellbeing AS wellbeing,
                   s.embedding AS embedding
            """
            nodes = self.run_query(node_query)
            nodes_list = [
                {
                    "id": record["id"],
                    "group": record.get("group", ""),
                    "achievement": record.get("achievement", None),
                    "wellbeing": record.get("wellbeing", None),
                    "embedding": record.get("embedding", [])
                }
                for record in nodes
            ]

            return {"nodes": nodes_list, "links": links}
        return {"nodes": [], "links": []}

    def get_student_relationships(self, student_id):
        """Get all relationships for a specific student."""
        query = """
        MATCH (s:Student {student_id: $student_id})-[r:RELATES_TO]->(s2:Student)
        RETURN s2.student_id as target_id,
               r.type as relationship_type,
               r.attention as attention
        """
        return self.run_query(query, {"student_id": student_id})

    def update_relationship(self, student1_id, student2_id, relationship_type, attention):
        """Update an existing relationship between students."""
        query = """
        MATCH (s1:Student {student_id: $student1_id})-[r:RELATES_TO]->(s2:Student {student_id: $student2_id})
        SET r.type = $relationship_type,
            r.attention = $attention
        """
        return self.run_query(query, {
            "student1_id": student1_id,
            "student2_id": student2_id,
            "relationship_type": relationship_type,
            "attention": attention
        })

    def delete_relationship(self, student1_id, student2_id):
        """Delete a relationship between students."""
        query = """
        MATCH (s1:Student {student_id: $student1_id})-[r:RELATES_TO]->(s2:Student {student_id: $student2_id})
        DELETE r
        """
        return self.run_query(query, {
            "student1_id": student1_id,
            "student2_id": student2_id
        })

# Create a singleton instance
neo4j_service = Neo4jService() 