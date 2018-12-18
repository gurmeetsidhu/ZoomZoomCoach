// Used for memory-mapped functionality
#include <windows.h>
#include "sharedmemory.h"
#include <fstream>
#include <stdlib.h>
using namespace std;

// Used for this example
#include <stdio.h>
#include <conio.h>

// Name of the pCars memory mapped file
#define MAP_OBJECT_NAME "$pcars2$"

int main()
{
	// Open the memory-mapped file
	HANDLE fileHandle = OpenFileMapping(PAGE_READONLY, FALSE, MAP_OBJECT_NAME);
	if (fileHandle == NULL)
	{
		printf("Could not open file mapping object (%d).\n", GetLastError());
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
	system("exec rm -r D:\\ScreenRecordings\\*");

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
		if ( isValidParticipantIndex && localCopy->mRaceState == 2 && localCopy->mGameState == 2)
		{
			//If not currently recording start fraps recording
			if (recording == FALSE) {
				ip.type = INPUT_KEYBOARD;
				ip.ki.wScan = 0; // hardware scan code for key
				ip.ki.time = 0;
				ip.ki.dwExtraInfo = 0;
				// Press the "A" key
				ip.ki.wVk = 0x78; // virtual-key code for the "a" key
				ip.ki.dwFlags = 0; // 0 for key press
				SendInput(1, &ip, sizeof(INPUT));
				Sleep(1000);
				// Release the "A" key
				ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
				SendInput(1, &ip, sizeof(INPUT));
				printf("Pressed F9 \n");
				recording = TRUE;
			}
			const ParticipantInfo& viewedParticipantInfo = localCopy->mParticipantInfo[sharedData->mViewedParticipantIndex];
			//printf( "WorldPosition: [%f,%f,%f]\n", viewedParticipantInfo.mWorldPosition[0], viewedParticipantInfo.mWorldPosition[1], viewedParticipantInfo.mWorldPosition[2]);
			//printf( "lap Distance = %f \n\n", viewedParticipantInfo.mCurrentLapDistance );

			//printf("mCurrentTime: %f\n", localCopy->mCurrentTime);
			//printf("mSpeed: %f\n", localCopy->mSpeed);
			//printf("mRpm: %f\n", localCopy->mRpm);
			//printf("mBrake: %f\n", localCopy->mBrake);
			//printf("mThrottle: %f\n", localCopy->mThrottle);
			//printf("mSteering: %f\n", localCopy->mSteering);
			//printf("mGear: %d\n", localCopy->mGear);
			//printf("mEngineTorque: %f\n", localCopy->mEngineTorque);

			//printf("mOrientation: [%f,%f,%f]\n", localCopy->mOrientation[0], localCopy->mOrientation[1], localCopy->mOrientation[2]);
			//printf("mLocalVelocity: [%f,%f,%f]\n", localCopy->mLocalVelocity[0], localCopy->mLocalVelocity[1], localCopy->mLocalVelocity[2]);
			//printf("mWorldVelocity: [%f,%f,%f]\n", localCopy->mWorldVelocity[0], localCopy->mWorldVelocity[1], localCopy->mWorldVelocity[2]);
			//printf("mAngularVelocity: [%f,%f,%f]\n", localCopy->mAngularVelocity[0], localCopy->mAngularVelocity[1], localCopy->mAngularVelocity[2]);
			//printf("mLocalAcceleration: [%f,%f,%f]\n", localCopy->mLocalAcceleration[0], localCopy->mLocalAcceleration[1], localCopy->mLocalAcceleration[2]);
			//printf("mWorldAcceleration: [%f,%f,%f]\n", localCopy->mWorldAcceleration[0], localCopy->mWorldAcceleration[1], localCopy->mWorldAcceleration[2]);
			//printf("mTerrain: [%d,%d,%d,%d]\n", localCopy->mTerrain[0], localCopy->mTerrain[1], localCopy->mTerrain[2], localCopy->mTerrain[3]);
			//printf("mTyreY: [%f,%f,%f,%f]\n", localCopy->mTyreY[0], localCopy->mTyreY[1], localCopy->mTyreY[2], localCopy->mTyreY[3]);
			//printf("mTyreRPS: [%f,%f,%f,%f]\n", localCopy->mTyreRPS[0], localCopy->mTyreRPS[1], localCopy->mTyreRPS[2], localCopy->mTyreRPS[3]);
			//printf("mTyreSlipSpeed: [%f,%f,%f,%f]\n", localCopy->mTyreSlipSpeed[0], localCopy->mTyreSlipSpeed[1], localCopy->mTyreSlipSpeed[2], localCopy->mTyreSlipSpeed[3]);
			//printf("mTyreGrip: [%f,%f,%f,%f]\n", localCopy->mTyreGrip[0], localCopy->mTyreGrip[1], localCopy->mTyreGrip[2], localCopy->mTyreGrip[3]);
			//printf("mTyreWear: [%f,%f,%f,%f]\n", localCopy->mTyreWear[0], localCopy->mTyreWear[1], localCopy->mTyreWear[2], localCopy->mTyreWear[3]);

			//Used for mapping
			//outFile << viewedParticipantInfo.mCurrentLap << "," << viewedParticipantInfo.mWorldPosition[0] << "," << viewedParticipantInfo.mWorldPosition[1] << "," << viewedParticipantInfo.mWorldPosition[2] << endl;

			//Used for telemetry data
			outFile << localCopy->mCarName << "," << localCopy->mTrackLocation << "," << localCopy->mTrackVariation << "," << viewedParticipantInfo.mCurrentLap << "," << viewedParticipantInfo.mWorldPosition[0] << "," << viewedParticipantInfo.mWorldPosition[1] << "," << viewedParticipantInfo.mWorldPosition[2] << "," << localCopy->mLocalAcceleration[0] << "," << localCopy->mLocalAcceleration[1] << "," << localCopy->mLocalAcceleration[2] << "," << localCopy->mSteering << "," << localCopy->mBrake << "," << localCopy->mThrottle << "," << localCopy->mTyreRPS[0] << "," << localCopy->mTyreRPS[1] << "," << localCopy->mTyreRPS[2] << "," << localCopy->mTyreRPS[3] << "," << localCopy->mWorldVelocity[0] << "," << localCopy->mWorldVelocity[1] << "," << localCopy->mWorldVelocity[2] << "," << localCopy->mOrientation[0] << "," << localCopy->mOrientation[1] << "," << localCopy->mOrientation[2] << "," << localCopy->mSpeed << "," << localCopy->mCurrentTime << "," << localCopy->mAngularVelocity[0] << "," << localCopy->mAngularVelocity[1] << "," << localCopy->mAngularVelocity[2] << endl;
		}
		else {
			//If game state changes stop recording. Change recording back to False so it can be restarted once player returns.
			if (recording == TRUE) {
				ip.type = INPUT_KEYBOARD;
				ip.ki.wScan = 0; // hardware scan code for key
				ip.ki.time = 0;
				ip.ki.dwExtraInfo = 0;
				// Press the "A" key
				ip.ki.wVk = 0x79; // virtual-key code for the "a" key
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
