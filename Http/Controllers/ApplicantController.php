<?php

namespace App\Http\Controllers;

use App\Models\QueueEntry;
use Illuminate\Http\Request;

class ApplicantController extends Controller
{
    // Show the customer queue form
    public function showForm()
    {
        $services = config('services_list.services');
        return view('applicant.form', compact('services'));
    }

    // Handle form submission and store queue entry
    public function submit(Request $request)
    {
        $data = $request->validate([
            'name' => 'required|string',
            'email' => 'required|email',
            'service' => 'required|string'
        ]);

       // Validate form input
$data = $request->validate([
    'name' => 'required|string',
    'email' => 'required|email',
    'service' => 'required|string',
]);

// Generate unique queue number
do {
    $queueNumber = 'Q-' . rand(1000, 9999);
} while (QueueEntry::where('queue_number', $queueNumber)->exists());

$data['queue_number'] = $queueNumber;
$data['status'] = 'waiting'; // Default status

// Save to database
$entry = QueueEntry::create($data);

return redirect('/status/' . $entry->id);
    }
    // Show status page for a specific queue entry
    public function status($id)
    {
        $entry = QueueEntry::findOrFail($id);
        $nowServing = QueueEntry::where('service', $entry->service)
                                ->where('status', 'in_service')
                                ->first();

        return view('applicant.status', compact('entry', 'nowServing'));
    }
}
