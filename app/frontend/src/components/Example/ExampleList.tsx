import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "the Green Edition2 카드로 여행 영역에서 받을 수 있는 M포인트 적립률은 얼마인가요?",
        value: "the Green Edition2 카드로 여행 영역에서 받을 수 있는 M포인트 적립률은 얼마인가요?"
    },
    {
        text: "the Pink와 the Green Edition2 카드는 모두 무료 발레파킹이 가능한가요?",
        value: "the Pink와 the Green Edition2 카드는 모두 무료 발레파킹이 가능한가요?"
    },
    {
        text: "이번 해 the Red Stripe 카드로 공항 라운지 총 8회 이용했는데, 이번 해에 몇 회 더 이용 가능한가요?",
        value: "이번 해 the Red Stripe 카드로 공항 라운지 총 8회 이용했는데, 이번 해에 몇 회 더 이용 가능한가요?"
    },
    {
        text: "파인 다이닝을 자주 이용하는 회원에게 추천할 만한 현대카드는 무엇인가요?",
        value: "파인 다이닝을 자주 이용하는 회원에게 추천할 만한 현대카드는 무엇인가요?"
    },
    {
        text: "골프를 많이 치고 바우처 혜택도 좋아하는 회원에게 추천할 만한 현대카드는 무엇인가요?",
        value: "골프를 많이 치고 바우처 혜택도 좋아하는 회원에게 추천할 만한 현대카드는 무엇인가요?"
    },
    {
        text: "올해 결혼할 예정인데 저에게 적합한 카드는 무엇인가요?",
        value: "올해 결혼할 예정인데 저에게 적합한 카드는 무엇인가요?"
    },
    {
        text: "the Purple osee의 Priority Pass 카드 혜택을 홍보하는 문구를 만들어주세요.",
        value: "the Purple osee의 Priority Pass 카드 혜택을 홍보하는 문구를 만들어주세요."
    }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
