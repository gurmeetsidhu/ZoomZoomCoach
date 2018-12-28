// Used for memory-mapped functionality
#include <windows.h>
#include "sharedmemory.h"
#include <fstream>
#include <stdlib.h>
#include <tlhelp32.h>
using namespace std;

// Used for this example
#include <stdio.h>
#include <conio.h>

// Name of the pCars memory mapped file
#define MAP_OBJECT_NAME "$pcars2$"

int main()
{
	//Check if OBS Studio is running. https://stackoverflow.com/questions/865152/how-can-i-get-a-process-handle-by-its-name-in-c
	PROCESSENTRY32 entry;
	entry.dwSize = sizeof(PROCESSENTRY32);
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, NULL);
	bool ObsOpen = FALSE;
	if (Process32First(snapshot, &entry) == TRUE) {
		while (Process32Next(snapshot, &entry) == TRUE) {
			if (stricmp(entry.szExeFile, "obs64.exe") == 0 || stricmp(entry.szExeFile, "obs32.exe")==0) {
				ObsOpen = TRUE;
			}
		}
	}
	if (ObsOpen == FALSE) {
		printf("OBS Studio not detected. Please open OBS studio if installed otherwise install. Ensure hotkeys are configured (Default F9/F10)\n");
		return 1;
	}

	// Open the memory-mapped file
	HANDLE fileHandle = OpenFileMapping(PAGE_READONLY, FALSE, MAP_OBJECT_NAME);
	if (fileHandle == NULL)
	{
		printf("Could not open file mapping object (%d). Please ensure PC2 is launched and has data sharing activated in game.\n", GetLastError());
		return 1;
	}

	// Get the data structure
	const SharedMemory* sharedData = (SharedMemory*)MapViewOfFile(fileHandle, PAGE_READONLY, 0, 0, sizeof(SharedMemory));
	SharedMemory* localCopy = new SharedMemory;
	if (sharedData == NULL)
	{
		printf("Could not map view of file (%d).\n", GetLastError());

		CloseHandle(fileHandle);
		return 1;
	}

	// Ensure we're sync'd to the correct data version
	if (sharedData->mVersion != SHARED_MEMORY_VERSION)
	{
		printf("Data version mismatch\n");
		printf("Game Shared Data Version: (%d)\n", sharedData->mVersion);
		printf("Code Shared Data Version: (%d)\n", SHARED_MEMORY_VERSION);
		return 1;
	}

	ofstream outFile;
	outFile.open("mapLog.csv");
	outFile << "mCarName" << "," << "mTrackLocation" << "," << "mTrackVariation" << "," << "mCurrentLap" << "," << "mWorldPosition[0]" << "," << "mWorldPosition[1]" << "," << "mWorldPosition[2]" << "," << "mLocalAcceleration[0]" << "," << "mLocalAcceleration[1]" << "," << "mLocalAcceleration[2]" << "," << "mSteering" << "," << "mBrake" << "," << "mThrottle" << "," << "mTyreRPS[0]" << "," << "mTyreRPS[1]" << "," << "mTyreRPS[2]" << "," << "mTyreRPS[3]" << "," << "mWorldVelocity[0]" << "," << "mWorldVelocity[1]" << "," << "mWorldVelocity[2]" << "," << "mOrientation[0]" << "," << "mOrientation[1]" << "," << "mOrientation[2]" << "," << "mSpeed" << "," << "mCurrentTime" << "," << "mAngularVelocity[0]" << "," << "mAngularVelocity[1]" << "," << "mAngularVelocity[2]" << endl;

	//------------------------------------------------------------------------------
	// TEST DISPLAY CODE
	//------------------------------------------------------------------------------
	unsigned int updateIndex(0);
	unsigned int indexChange(0);

	printf( "ESC TO EXIT\n\n" );
	INPUT ip;
	WORD vkey = VK_F9;
	WORD vkey2 = VK_F10;
	bool recording = FALSE;

	while (true)
	{
		if ( sharedData->mSequenceNumber % 2 )
		{
			// Odd sequence number indicates, that write into the shared memory is just happening
			continue;
		}

		indexChange = sharedData->mSequenceNumber - updateIndex;
		updateIndex = sharedData->mSequenceNumber;

		//Copy the whole structure before processing it, otherwise the risk of the game writing into it during processing is too high.
		memcpy(localCopy,sharedData,sizeof(SharedMemory));


		if (localCopy->mSequenceNumber != updateIndex )
		{
			// More writes had happened during the read. Should be rare, but can happen.
			continue;
		}

		printf( "Sequence number increase %d, current index %d, previous index %d\n", indexChange, localCopy->mSequenceNumber, updateIndex );

		const bool isValidParticipantIndex = localCopy->mViewedParticipantIndex != -1 && localCopy->mViewedParticipantIndex < localCopy->mNumParticipants && localCopy->mViewedParticipantIndex < STORED_PARTICIPANTS_MAX;
		if ( isValidParticipantIndex && localCopy->mRaceState == 2 && localCopy->mGameState == 2) //Ensure we are racing before starting recording
		{
			//If not currently recording start fraps recording
			const ParticipantInfo& viewedParticipantInfo = localCopy->mParticipantInfo[sharedData->mViewedParticipantIndex];
			if (recording == FALSE) {
				ip.type = INPUT_KEYBOARD;
				ip.ki.wScan = 0; // hardware scan code for key
				ip.ki.time = 0;
				ip.ki.dwExtraInfo = 0;
				// Press the "A" key
				ip.ki.wVk = 0x78; // virtual-key code for the "F9" key
				ip.ki.dwFlags = 0; // 0 for key press
				SendInput(1, &ip, sizeof(INPUT));
				Sleep(50);
				// Release the "A" key
				ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
				SendInput(1, &ip, sizeof(INPUT));
				recording = TRUE;
			}
			else {
				outFile << localCopy->mCarName << "," << localCopy->mTrackLocation << "," << localCopy->mTrackVariation << "," << viewedParticipantInfo.mCurrentLap << "," << viewedParticipantInfo.mWorldPosition[0] << "," << viewedParticipantInfo.mWorldPosition[1] << "," << viewedParticipantInfo.mWorldPosition[2] << "," << localCopy->mLocalAcceleration[0] << "," << localCopy->mLocalAcceleration[1] << "," << localCopy->mLocalAcceleration[2] << "," << localCopy->mSteering << "," << localCopy->mBrake << "," << localCopy->mThrottle << "," << localCopy->mTyreRPS[0] << "," << localCopy->mTyreRPS[1] << "," << localCopy->mTyreRPS[2] << "," << localCopy->mTyreRPS[3] << "," << localCopy->mWorldVelocity[0] << "," << localCopy->mWorldVelocity[1] << "," << localCopy->mWorldVelocity[2] << "," << localCopy->mOrientation[0] << "," << localCopy->mOrientation[1] << "," << localCopy->mOrientation[2] << "," << localCopy->mSpeed << "," << localCopy->mCurrentTime << "," << localCopy->mAngularVelocity[0] << "," << localCopy->mAngularVelocity[1] << "," << localCopy->mAngularVelocity[2] << endl;
			}
		}
		else {
			//If game state changes stop recording. Change recording back to False so it can be restarted once player returns.
			if (recording == TRUE) {
				ip.type = INPUT_KEYBOARD;
				ip.ki.wScan = 0; // hardware scan code for key
				ip.ki.time = 0;
				ip.ki.dwExtraInfo = 0;
				// Press the "A" key
				ip.ki.wVk = 0x79; // virtual-key code for the "F10" key
				ip.ki.dwFlags = 0; // 0 for key press
				SendInput(1, &ip, sizeof(INPUT));
				Sleep(1000);
				// Release the "A" key
				ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
				SendInput(1, &ip, sizeof(INPUT));
				printf("Stopped Recording Pressed F10 \n");
				recording = FALSE;
			}
		}

		system("cls");

		if ( _kbhit() && _getch() == 27 ) // check for escape
		{
			break;
		}
	}
	//------------------------------------------------------------------------------

	// Cleanup
	UnmapViewOfFile( sharedData );
	CloseHandle( fileHandle );
	delete localCopy;

	outFile.close();

	return 0;
}
