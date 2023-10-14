CREATE TABLE bitva (
    bitva_nazev VARCHAR2(50) NOT NULL,
    bitva_datum VARCHAR2(20) NOT NULL,
    bitva_popis VARCHAR2(200)
);

ALTER TABLE bitva ADD CONSTRAINT bitva_pk PRIMARY KEY ( bitva_nazev );

GRANT SELECT ON bitva TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON bitva TO DB4IT218;


CREATE TABLE jedi (
    jedi_cis         INTEGER NOT NULL
        CONSTRAINT jedi_cis_jedi CHECK (jedi_cis > 0),
    jedi_jmeno       VARCHAR2(50) NOT NULL,
    jedi_narozeniny  VARCHAR2(20) NOT NULL,
    jedi_druh        VARCHAR2(50) NOT NULL,
    lightsaber_color VARCHAR2(40)
);

ALTER TABLE jedi ADD CONSTRAINT jedi_pk PRIMARY KEY ( jedi_cis );

GRANT SELECT ON jedi TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON jedi TO DB4IT218;



CREATE TABLE klon (
    klon_cis     VARCHAR2(20) NOT NULL,
    klon_variant VARCHAR2(50) NOT NULL,
    volaci_jmeno VARCHAR2(50)
);

ALTER TABLE klon ADD CONSTRAINT klon_pk PRIMARY KEY ( klon_cis );

GRANT SELECT ON klon TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON klon TO DB4IT218;



CREATE TABLE participace (
    unit_unit_cis     INTEGER NOT NULL,
    bitva_bitva_nazev VARCHAR2(50) NOT NULL
);

ALTER TABLE participace ADD CONSTRAINT participace_pk PRIMARY KEY ( unit_unit_cis,
                                                                    bitva_bitva_nazev );
                                                                    
GRANT SELECT ON participace TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON participace TO DB4IT218;

                                                                    

CREATE TABLE unit (
    unit_cis      INTEGER NOT NULL
        CONSTRAINT unit_cis_unit CHECK (unit_cis > 0),
    unit_nazev    VARCHAR2(50) NOT NULL,
    unit_color    VARCHAR2(40),
    klon_klon_cis VARCHAR2(20)
);

CREATE UNIQUE INDEX unit__idx ON
    unit (
        klon_klon_cis
    ASC );

ALTER TABLE unit ADD CONSTRAINT unit_pk PRIMARY KEY ( unit_cis );

GRANT SELECT ON unit TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON unit TO DB4IT218;



CREATE TABLE vedeni (
    jedi_jedi_cis INTEGER NOT NULL,
    unit_unit_cis INTEGER NOT NULL
);

ALTER TABLE vedeni ADD CONSTRAINT vedeni_pk PRIMARY KEY ( jedi_jedi_cis,
                                                          unit_unit_cis );
                                                          
GRANT SELECT ON vedeni TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON vedeni TO DB4IT218;

                                                          

CREATE TABLE zarazeni (
    klon_klon_cis VARCHAR2(20) NOT NULL,
    unit_unit_cis INTEGER NOT NULL
);

ALTER TABLE zarazeni ADD CONSTRAINT zarazeni_pk PRIMARY KEY ( klon_klon_cis,
                                                              unit_unit_cis );
                                                              
GRANT SELECT ON zarazeni TO STUDENT;
GRANT DELETE,INSERT,SELECT,UPDATE ON zarazeni TO DB4IT218;                                                              
                                                              


ALTER TABLE participace
    ADD CONSTRAINT participace_bitva_fk FOREIGN KEY ( bitva_bitva_nazev )
        REFERENCES bitva ( bitva_nazev );

ALTER TABLE participace
    ADD CONSTRAINT participace_unit_fk FOREIGN KEY ( unit_unit_cis )
        REFERENCES unit ( unit_cis );

ALTER TABLE unit
    ADD CONSTRAINT unit_klon_fk FOREIGN KEY ( klon_klon_cis )
        REFERENCES klon ( klon_cis )
            ON DELETE CASCADE;

ALTER TABLE vedeni
    ADD CONSTRAINT vedeni_jedi_fk FOREIGN KEY ( jedi_jedi_cis )
        REFERENCES jedi ( jedi_cis );

ALTER TABLE vedeni
    ADD CONSTRAINT vedeni_unit_fk FOREIGN KEY ( unit_unit_cis )
        REFERENCES unit ( unit_cis );

ALTER TABLE zarazeni
    ADD CONSTRAINT zarazeni_klon_fk FOREIGN KEY ( klon_klon_cis )
        REFERENCES klon ( klon_cis );

ALTER TABLE zarazeni
    ADD CONSTRAINT zarazeni_unit_fk FOREIGN KEY ( unit_unit_cis )
        REFERENCES unit ( unit_cis );



-- Oracle SQL Developer Data Modeler Summary Report: 
-- 
-- CREATE TABLE                             7
-- CREATE INDEX                             1
-- ALTER TABLE                             14
-- CREATE VIEW                              0
-- ALTER VIEW                               0
-- CREATE PACKAGE                           0
-- CREATE PACKAGE BODY                      0
-- CREATE PROCEDURE                         0
-- CREATE FUNCTION                          0
-- CREATE TRIGGER                           0
-- ALTER TRIGGER                            0
-- CREATE COLLECTION TYPE                   0
-- CREATE STRUCTURED TYPE                   0
-- CREATE STRUCTURED TYPE BODY              0
-- CREATE CLUSTER                           0
-- CREATE CONTEXT                           0
-- CREATE DATABASE                          0
-- CREATE DIMENSION                         0
-- CREATE DIRECTORY                         0
-- CREATE DISK GROUP                        0
-- CREATE ROLE                              0
-- CREATE ROLLBACK SEGMENT                  0
-- CREATE SEQUENCE                          0
-- CREATE MATERIALIZED VIEW                 0
-- CREATE MATERIALIZED VIEW LOG             0
-- CREATE SYNONYM                           0
-- CREATE TABLESPACE                        0
-- CREATE USER                              0
-- 
-- DROP TABLESPACE                          0
-- DROP DATABASE                            0
-- 
-- REDACTION POLICY                         0
-- 
-- ORDS DROP SCHEMA                         0
-- ORDS ENABLE SCHEMA                       0
-- ORDS ENABLE OBJECT                       0
-- 
-- ERRORS                                   0
-- WARNINGS                                 0
