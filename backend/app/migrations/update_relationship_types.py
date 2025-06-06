from app.services.neo4j_service import neo4j_service, RELATIONSHIP_TYPES

def migrate_relationship_types():
    """Update the database schema to support relationship types"""
    # Create constraints
    neo4j_service.create_constraints()
    
    # Create relationship type property constraint
    query = """
    CREATE CONSTRAINT IF NOT EXISTS FOR ()-[r:RELATES_TO]-() 
    REQUIRE (r.type, r.attention) IS UNIQUE
    """
    neo4j_service.run_query(query)
    
    # Update existing relationships to use correct types
    query = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE NOT r.type IN ['friend', 'neutral', 'conflict']
    SET r.type = CASE
        WHEN r.type IN ['study_partner', 'mentor', 'project', 'family', 'sports'] THEN 'friend'
        WHEN r.type IN ['rival', 'avoid'] THEN 'conflict'
        ELSE 'neutral'
    END
    """
    neo4j_service.run_query(query)
    
    print("✅ Successfully migrated relationship types")

if __name__ == "__main__":
    migrate_relationship_types() 