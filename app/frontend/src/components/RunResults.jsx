import {
    Section,
    SectionCard,
    Card,
    CardList,
    Label,
    Text,
} from "@blueprintjs/core";

// result format:
//        results = {
//            "model": self.name,
//            "parameters": self.parameters,
//            "start_time": str(start_time),
//            "end_time": str(end_time),
//            "total_duration": total_duration,
//            "results": {
//                "accuracy": accuracy,
//                "precision": precision,
//                "recall": recall,
//                "avg_f1": avg_f1,
//                "categorical_f1": f1.tolist(),
//            },
//        }

function RunResults({ modelOutput }) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
            }}
        >
            <h2>Results</h2>
            {modelOutput && (
                <>
                    <Text>Start time: {modelOutput.start_time}</Text>
                    <Text>End time: {modelOutput.end_time}</Text>
                    <Text>Total Duration: {modelOutput.total_duration} seconds</Text>
                </>
            )}
            {modelOutput &&
                <CardList bordered={false}>
                    <Card>
                        Accuracy: {modelOutput.results.accuracy}
                    </Card>
                    <Card>
                        Precision: {modelOutput.results.precision}
                    </Card>
                    <Card>
                        Recall: {modelOutput.results.recall}
                    </Card>
                    <Card>
                        Average F1: {modelOutput.results.avg_f1}
                    </Card>
                    <Card>
                        Categorical F1: {modelOutput.results.categorical_f1.join(", ")}
                    </Card>

                </CardList>
            }
            {!modelOutput && <p>No results to display</p>}
        </div >
    );
}

export default RunResults;
