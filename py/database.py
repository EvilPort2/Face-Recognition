import sqlite3
import os


def displayDbContent():
    if os.name == 'nt':
        os.system("cls")
    else:
        os.system("clear")
    conn = sqlite3.connect("faceDetectDatabase.db")
    cmd = "SELECT * FROM People"
    cursor = conn.execute(cmd)
    print ("ID\t\t\tName\t\t\t\tOccupation\t\t\tGender")
    print (95 * "-")
    for row in cursor:
        print("%5s%30s%30s%30s" % (str(row[0]), str(row[1]), str(row[2]), str(row[3])))
        
    print ("\n\n")
    conn.close()


def getProfileDataById(id):
    conn = sqlite3.connect("faceDetectDatabase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(id)
    cursor = conn.execute(cmd)
    for row in cursor:
        profile = row
    return profile
